"""Prism pretrain — TimesFM 2.5 freeze + 5 ResidualBlock decoders
(trend × 1, seasonal × 3 [lo/mid/hi], residual × 1) on LOTSA + Monash.

5-stage stage-wise schedule:
  stage 1: trend
  stage 2: seasonal_lo  (target = window − wt)
  stage 3: seasonal_mid (target = window − wt − ws_lo)
  stage 4: seasonal_hi  (target = window − wt − ws_lo − ws_mid)
  stage 5: residual     (target = window − wt − ws_lo − ws_mid − ws_hi)

Loss: per-patch teacher-forcing MAE in normalized space, masking invalid
patches (left-pad or random-mask augmented). Same recipe as april's `april_resi`
training, but extended to 5 stages and ctx_len=1024.

Default scale: ~16M training windows total
  = 5 stages × 6250 steps × batch 512 = 16M  (batch 512 chosen for ctx=1024 mem)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]   # /home/chaewon/projects/timesfm
sys.path.insert(0, str(PROJECT_ROOT))

from prism.model.decomp_timesfm_prism import DecompTimesFMPrism  # noqa: E402
from prism.data.pretrain_loader import (                          # noqa: E402
    build_dataloader, enumerate_monash, build_subsets,
)
from timesfm.torch.util import revin                              # noqa: E402

OUTPUT_DIR = HERE
CKPT_DIR = OUTPUT_DIR / "ckpts"
CKPT_DIR.mkdir(exist_ok=True)

DEFAULT_CFG = {
    # Backbone-aligned
    "context_len": 1024,
    "horizon": 128,
    "patch_len": 32,
    "embed_dim": 1280,
    # Decoder freq decomposition
    "n_freq_downsample_trend": 32,
    "n_freq_downsample_seasonal": [16, 8, 4],
    # ResidualBlock head config
    "hidden_dim": 1280,
    "use_bias": False,
    "activation": "swish",
    # Optimizer / regularization
    "learning_rate": 1e-3,
    "batch_size": 512,
    # Mask augmentation
    "mask_aug_prob": 0.5,
    "mask_aug_max_patches": 16,   # ctx=1024 / patch=32 = 32 patches; mask up to half
}

# Stage 1..5 → which decoder is unfrozen
STAGE_NAMES = {
    1: "trend",
    2: "seasonal_lo",
    3: "seasonal_mid",
    4: "seasonal_hi",
    5: "residual",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_per_stage", type=int, default=6250,
                    help="5 stages × 6250 × batch 512 ≈ 16M windows")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_aug_prob", type=float, default=0.5)
    ap.add_argument("--mask_aug_max_patches", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--smoke", type=int, default=0,
                    help="If > 0, run this many steps per stage and exit")
    ap.add_argument("--corpus", type=str, default="all",
                    choices=["monash", "all"],
                    help="all = LOTSA + Monash; monash = Monash only")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def _decoder_state_dict(model):
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if not k.startswith("backbone.")}


def save_ckpt(model, stage: int, step: int, cfg: dict, losses_history: list):
    path = CKPT_DIR / f"stage{stage}_step{step:07d}.pt"
    torch.save({
        "stage": stage, "step_in_stage": step,
        "decoder_state_dict": _decoder_state_dict(model),
        "cfg": cfg, "losses_history": losses_history,
    }, path)
    print(f"  [ckpt] saved {path.name}")
    return path


def load_latest_ckpt():
    cands = []
    for p in CKPT_DIR.glob("stage*_step*.pt"):
        try:
            stage = int(p.stem.split("_")[0][5:])
            step = int(p.stem.split("_")[1][4:])
            cands.append(((stage, step), p))
        except Exception:
            continue
    if not cands:
        return None, None
    cands.sort()
    p = cands[-1][1]
    return p, torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Per-patch teacher-forcing target builder + random-mask augmentation
# ---------------------------------------------------------------------------

def build_targets(window: torch.Tensor, p: int, h: int):
    B, L = window.shape
    P = (L - h) // p
    out = window.unfold(dimension=1, size=h, step=p)[:, 1: P + 1, :]
    return out, P


def apply_random_left_mask(masks: torch.Tensor, P_input: int, prob: float,
                           max_patches: int, generator: torch.Generator | None = None):
    if prob <= 0 or max_patches <= 0:
        return masks
    B, L = masks.shape
    p = L // P_input
    device = masks.device
    coin = torch.rand(B, device=device, generator=generator)
    do_mask = coin < prob
    if not do_mask.any():
        return masks
    Ks = torch.randint(1, max_patches + 1, (B,), device=device, generator=generator)
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    threshold = (Ks * p).unsqueeze(1)
    new_mask = (pos < threshold) & do_mask.unsqueeze(1)
    return masks | new_mask


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

DECODER_ATTR = {
    1: "decoder_t",
    2: "decoder_s_lo",
    3: "decoder_s_mid",
    4: "decoder_s_hi",
    5: "decoder_r",
}


def set_stage_requires_grad(model, stage: int):
    for s, attr in DECODER_ATTR.items():
        for p in getattr(model, attr).parameters():
            p.requires_grad = (s == stage)


def stage_pred_and_target(stage: int, out: dict, targets_n: torch.Tensor):
    """Stage-specific prediction and (residual) target.
    Each stage decodes the residual after subtracting prior decoders' outputs."""
    wt = out["wave_t_n"]
    ws_lo = out["wave_s_lo_n"]
    ws_mid = out["wave_s_mid_n"]
    ws_hi = out["wave_s_hi_n"]
    wr = out["wave_r_n"]

    if stage == 1:
        return wt, targets_n
    if stage == 2:
        target = (targets_n - wt.detach()).detach()
        return ws_lo, target
    if stage == 3:
        target = (targets_n - wt.detach() - ws_lo.detach()).detach()
        return ws_mid, target
    if stage == 4:
        target = (targets_n - wt.detach() - ws_lo.detach()
                  - ws_mid.detach()).detach()
        return ws_hi, target
    if stage == 5:
        target = (targets_n - wt.detach() - ws_lo.detach()
                  - ws_mid.detach() - ws_hi.detach()).detach()
        return wr, target
    raise ValueError(stage)


def train_one_stage(model, stage, loader, args, cfg, device,
                    start_step: int = 0, losses_history: list | None = None):
    total_steps = args.steps_per_stage if not args.smoke else args.smoke
    set_stage_requires_grad(model, stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(0, min(args.warmup_steps, total_steps - 1))
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    for _ in range(start_step):
        scheduler.step()

    p = cfg["patch_len"]
    h = cfg["horizon"]
    ctx_len = cfg["context_len"]
    P_input = ctx_len // p

    losses = losses_history if losses_history is not None else []
    print(f"\n=== Stage {stage} ({STAGE_NAMES[stage]}) : {total_steps} steps, "
          f"start={start_step}, mask_aug={args.mask_aug_prob} ===")
    model.train()
    model.backbone.eval()

    it = iter(loader)
    step = start_step
    t0 = time.time()
    last_log_step, last_log_time = step, t0
    while step < total_steps:
        try:
            window, pad_mask = next(it)
        except StopIteration:
            it = iter(loader)
            window, pad_mask = next(it)
        window = window.to(device, non_blocking=True)        # [B, MAX_WL=1280]
        pad_mask = pad_mask.to(device, non_blocking=True)
        # Slice last (ctx+h)=1152 — real data right-aligned, pad on left.
        window = window[:, -(ctx_len + h):]
        pad_mask = pad_mask[:, -(ctx_len + h):]
        ctx = window[:, :ctx_len]
        masks = pad_mask[:, :ctx_len].clone()
        masks = apply_random_left_mask(
            masks, P_input,
            prob=args.mask_aug_prob,
            max_patches=args.mask_aug_max_patches,
        )

        out = model.forward_per_patch(ctx, masks)
        patch_mu = out["patch_mu"]
        patch_sigma = out["patch_sigma"]

        targets_raw, P = build_targets(window, p, h)
        targets_n = revin(targets_raw, patch_mu[:, :P], patch_sigma[:, :P], reverse=False)

        patched_input_mask_any = masks.reshape(window.shape[0], P_input, p).any(-1)
        target_pad = pad_mask.unfold(1, h, p)[:, 1: P + 1, :].any(-1)
        valid = ~(patched_input_mask_any[:, :P] | target_pad)

        pred_n, target = stage_pred_and_target(stage, out, targets_n)
        per_patch_huber = F.huber_loss(pred_n, target, reduction="none", delta=1.0).mean(dim=-1)
        valid_f = valid.float()
        loss = (per_patch_huber * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        step += 1

        if step == start_step + 1 or step % args.log_every == 0:
            dt = time.time() - last_log_time
            rate = (step - last_log_step) / max(dt, 1e-6)
            last_log_step, last_log_time = step, time.time()
            valid_frac = float(valid_f.mean().item())
            print(f"  stage={stage}({STAGE_NAMES[stage]}) step={step}/{total_steps} "
                  f"loss={loss.item():.6f} valid={valid_frac:.2f} "
                  f"rate={rate:.1f}/s elapsed={time.time()-t0:.0f}s")

        if step % args.ckpt_every == 0 and step < total_steps:
            save_ckpt(model, stage, step, cfg, losses)

    return losses


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = dict(DEFAULT_CFG)
    cfg["batch_size"] = args.batch_size
    cfg["learning_rate"] = args.lr
    cfg["mask_aug_prob"] = args.mask_aug_prob
    cfg["mask_aug_max_patches"] = args.mask_aug_max_patches
    cfg["steps_per_stage"] = args.steps_per_stage if not args.smoke else args.smoke
    cfg["corpus"] = args.corpus
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("Building DecompTimesFMPrism (backbone frozen)...")
    model = DecompTimesFMPrism(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Decoders trainable params: {n_trainable:,}")

    resume_stage = 1
    resume_step = 0
    existing = []
    if not args.smoke:
        ckpt_path, state = load_latest_ckpt()
        if ckpt_path is not None:
            model.load_state_dict(state["decoder_state_dict"], strict=False)
            resume_stage = int(state["stage"])
            resume_step = int(state["step_in_stage"])
            existing = state.get("losses_history", [])
            print(f"  Resumed from {ckpt_path.name}: stage={resume_stage} step={resume_step}")
            if resume_step >= args.steps_per_stage:
                resume_step = 0
                resume_stage += 1
                existing = []

    print(f"Building pretrain dataloader (corpus={args.corpus})...")
    subsets = enumerate_monash() if args.corpus == "monash" else build_subsets()
    loader, _ = build_dataloader(
        batch_size=cfg["batch_size"],
        num_workers=args.num_workers, subsets=subsets, seed=args.seed,
    )
    print(f"  Subsets used: {len(subsets)}")

    for stage in [1, 2, 3, 4, 5]:
        if stage < resume_stage:
            continue
        start_step = resume_step if stage == resume_stage else 0
        history = existing if stage == resume_stage else []
        losses = train_one_stage(model, stage, loader, args, cfg, device,
                                 start_step, history)
        save_ckpt(model, stage,
                  args.steps_per_stage if not args.smoke else args.smoke, cfg, losses)
        torch.save(
            {"decoder_state_dict": _decoder_state_dict(model), "cfg": cfg,
             "stage": stage, "final": True},
            OUTPUT_DIR / f"stage{stage}_final.pt",
        )

    torch.save(
        {"decoder_state_dict": _decoder_state_dict(model), "cfg": cfg},
        OUTPUT_DIR / "prism_pretrain_final.pt",
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
