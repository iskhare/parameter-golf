#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import time
from datetime import timedelta
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist
import zstandard


MODULE_PATH = Path("/workspace/parameter-golf/tmp/gdn_frontier/train_gpt.py")


def load_gdn_module():
    spec = importlib.util.spec_from_file_location("gdn_train_gpt", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and quantize a saved GDN EMA checkpoint.")
    parser.add_argument("--model-path")
    parser.add_argument("--artifact-path")
    parser.add_argument("--run-id", default="gdn_eval_only")
    parser.add_argument("--ckpt-dir", default="/workspace/parameter-golf/tmp/gdn_frontier/eval_artifacts")
    parser.add_argument("--arch-mode", default="D")
    parser.add_argument("--data-path", default="/workspace/parameter-golf/data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--eval-stride", type=int, default=64)
    parser.add_argument("--gptq-enabled", type=int, default=1)
    parser.add_argument("--gptq-num-seqs", type=int, default=64)
    parser.add_argument("--gptq-batch-size", type=int, default=8)
    parser.add_argument("--gptq-temperature", type=float, default=0.8)
    parser.add_argument("--qk-gain-init", type=float, default=0.0)
    parser.add_argument("--swa-window", type=int, default=0)
    parser.add_argument("--bigram-vocab-size", type=int, default=0)
    parser.add_argument("--bigram-dim", type=int, default=0)
    parser.add_argument("--trigram", type=int, default=-1)
    parser.add_argument("--mlp-mult", type=float, default=0.0)
    parser.add_argument("--layer-layout", default="")
    parser.add_argument("--skip-ema-eval", action="store_true")
    parser.add_argument("--skip-xsa-eval", action="store_true")
    parser.add_argument("--known-ema-bpb", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_path and not args.artifact_path:
        raise SystemExit("Provide --model-path or --artifact-path")
    mod = load_gdn_module()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_process = rank == 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(hours=1),
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log_dir = Path("/workspace/parameter-golf/tmp/gdn_frontier/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{args.run_id}.txt"

    def log0(msg: str) -> None:
        if not master_process:
            return
        print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as handle:
            print(msg, file=handle)

    config = mod.get_config(args.arch_mode)
    if args.qk_gain_init > 0:
        config["qk_gain_init"] = args.qk_gain_init
    if args.swa_window > 0:
        config["swa_window"] = args.swa_window
    if args.bigram_vocab_size > 0:
        config["bigram_vocab_size"] = args.bigram_vocab_size
    if args.bigram_dim > 0:
        config["bigram_dim"] = args.bigram_dim
    if args.trigram >= 0:
        config["trigram"] = bool(args.trigram)
    if args.mlp_mult > 0:
        config["mlp_mult"] = args.mlp_mult
    if args.layer_layout:
        config["layer_layout"] = args.layer_layout

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = mod.load_validation_tokens(
        os.path.join(args.data_path, "fineweb_val_*.bin"),
        args.eval_seq_len,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = mod.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    def build_model():
        mdl = mod.HybridGDN(config, args.vocab_size).to(device).bfloat16()
        for module in mdl.modules():
            if isinstance(module, mod.CastedLinear):
                module.float()
        for _, param in mdl.named_parameters():
            if param.ndim <= 1:
                param.data = param.data.float()
        return mdl

    model = build_model()
    log0(f"Validation tokens: {val_tokens.numel()-1:,}")

    if args.model_path:
        state_dict = torch.load(args.model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        log0(f"Loaded EMA checkpoint: {args.model_path}")

        if args.skip_ema_eval and args.known_ema_bpb > 0:
            ema_bpb = args.known_ema_bpb
            log0(f"EMA BPB (no XSA): {ema_bpb:.6f} [provided]")
        else:
            _, ema_bpb = mod.eval_val_sliding(
                model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                rank, world_size, device,
                seq_len=args.eval_seq_len, stride=args.eval_stride,
                xsa_eval=False,
            )
            log0(f"EMA BPB (no XSA): {ema_bpb:.6f}")
    else:
        ema_bpb = args.known_ema_bpb
        log0(f"Skipping EMA checkpoint load; artifact-only evaluation")
        if ema_bpb > 0:
            log0(f"EMA BPB (no XSA): {ema_bpb:.6f} [provided]")

    artifact_path = Path(args.artifact_path) if args.artifact_path else ckpt_dir / f"{Path(args.model_path).stem}.int6.ptz"
    if not args.artifact_path and master_process:
        hessians = None
        if args.gptq_enabled:
            log0("Generating GPTQ calibration sequences")
            calib_seqs = mod.generate_autoregressive_calib(
                model,
                device,
                num_seqs=args.gptq_num_seqs,
                seq_len=args.eval_seq_len,
                vocab_size=args.vocab_size,
                temperature=args.gptq_temperature,
                batch_size=args.gptq_batch_size,
                seed=args.seed,
            )
            log0(f"Collected {len(calib_seqs)} calibration sequences")
            hessians = mod.collect_hessians_from_tokens(model, calib_seqs, device)
            log0(f"Collected hessians for {len(hessians)} layers")

        sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        quant_result, quant_meta = mod.mixed_quantize(sd_cpu, hessians=hessians)
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_buf.getvalue())
        artifact_path.write_bytes(quant_blob)
        log0(f"Artifact: {len(quant_blob):,} bytes")

    if distributed:
        dist.barrier()

    quant_blob_disk = artifact_path.read_bytes()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk)),
        map_location="cpu",
        weights_only=False,
    )
    template_sd = {k: v.detach().cpu() for k, v in build_model().state_dict().items()}
    deq_state = mod.dequantize_mixed(quant_state["w"], quant_state["m"], template_sd)

    eval_model = build_model()
    eval_model.load_state_dict(deq_state, strict=True)

    quant_loss, quant_bpb = mod.eval_val_sliding(
        eval_model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, device,
        seq_len=args.eval_seq_len, stride=args.eval_stride,
        xsa_eval=False,
    )
    log0(f"Quantized BPB (no XSA): {quant_bpb:.6f}")
    log0(f"Quantization degradation: {quant_bpb - ema_bpb:+.6f}")

    if not args.skip_xsa_eval and any(bt in ("swa", "swa_shared") for bt in eval_model._block_types):
        xsa_loss, xsa_bpb = mod.eval_val_sliding(
            eval_model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            rank, world_size, device,
            seq_len=args.eval_seq_len, stride=args.eval_stride,
            xsa_eval=True,
        )
        log0(f"Quantized BPB (XSA-all): {xsa_bpb:.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
