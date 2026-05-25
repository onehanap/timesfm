"""Decompose training-distribution windows with the trained Prism model.

Prism = TimesFM 2.5 frozen backbone + 5 ResidualBlock decoders:
  trend (4 linear anchors), seasonal_lo (8), seasonal_mid (16), seasonal_hi (32)
  cubic anchors, and full-horizon residual.

Same windowing as the trainer:
    window_len = ctx (1024) + 2 * horizon (128) = 1280
We feed the first 1024 as context, treat the next 128 as GT future. Single
forward returns last-patch decompositions.

CPU-friendly: pass --device cpu.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]  # .../timesfm
APRIL_EXP_DIR = PROJECT_ROOT / "april" / "experiments"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APRIL_EXP_DIR))

from _common.pretrain_loader import MonashSubset, MONASH_ZIP_DIR  # noqa: E402


SUBSETS = [
    ("australian_electricity_demand_dataset", "half-hourly"),
    ("kdd_cup_2018_dataset_without_missing_values", "hourly"),
    ("nn5_daily_dataset_without_missing_values", "daily"),
]

DEFAULT_CKPT = str(PROJECT_ROOT / "april" / "experiments" / "pretrain" / "prism" / "prism_pretrain_final.pt")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=str,
                    default=str(HERE / "eval" / "decomp_plots" / "train_decomp_prism.png"))
    ap.add_argument("--per_subset", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_prism(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    from prism.model.decomp_timesfm_prism import DecompTimesFMPrism
    model = DecompTimesFMPrism(cfg).to(device)
    model.load_state_dict(state["decoder_state_dict"], strict=False)
    model.eval()
    return model, cfg


@torch.no_grad()
def prism_forward(model, ctx_np: np.ndarray, mask_np: np.ndarray, device):
    ctx = torch.from_numpy(ctx_np[None, :].astype(np.float32)).to(device)
    msk = torch.from_numpy(mask_np[None, :].astype(bool)).to(device)
    pred, decomp = model(ctx, msk)
    return {
        "pred": pred[0].cpu().numpy(),
        "trend": decomp["trend"][0].cpu().numpy(),
        "s_lo": decomp["seasonal_lo"][0].cpu().numpy(),
        "s_mid": decomp["seasonal_mid"][0].cpu().numpy(),
        "s_hi": decomp["seasonal_hi"][0].cpu().numpy(),
        "s_total": decomp["seasonal"][0].cpu().numpy(),
        "residual": decomp["residual"][0].cpu().numpy(),
    }


def sample_windows(subset_basename, n, ctx_len, horizon, patch_len, rng):
    zip_path = Path(MONASH_ZIP_DIR) / f"{subset_basename}.zip"
    sub = MonashSubset(str(zip_path))
    win_len = ctx_len + 2 * horizon
    min_len = patch_len + horizon
    candidates = [i for i in range(sub.n_series) if sub.get_length(i) >= min_len]
    if len(candidates) < n:
        raise RuntimeError(
            f"{subset_basename}: only {len(candidates)} series ≥ {min_len}"
        )
    chosen = rng.choice(candidates, size=n, replace=False)
    samples = []
    for seri in chosen:
        s = sub.get_series(int(seri))
        L = s.size
        if L >= win_len:
            max_start = L - win_len
            start = int(rng.integers(0, max_start + 1))
            real = s[start : start + win_len].astype(np.float32)
            real_len = win_len
            padded = False
        else:
            start = 0
            real = s.astype(np.float32)
            real_len = L
            padded = True

        ctx_buf = np.zeros(ctx_len, dtype=np.float32)
        msk_buf = np.zeros(ctx_len, dtype=bool)
        future = np.full(horizon, np.nan, dtype=np.float32)
        if real_len >= ctx_len + horizon:
            ctx_buf[:] = real[:ctx_len]
            future[:] = real[ctx_len : ctx_len + horizon]
        else:
            future[:] = real[-horizon:]
            real_ctx = real[:-horizon]
            n_pad = ctx_len - real_ctx.size
            ctx_buf[n_pad:] = real_ctx
            msk_buf[:n_pad] = True

        samples.append({
            "subset": sub.name,
            "freq": sub.frequency or "?",
            "series_idx": int(seri),
            "start": start,
            "ctx": ctx_buf,
            "mask": msk_buf,
            "real_ctx_len": int((~msk_buf).sum()),
            "future": future,
            "padded": padded,
        })
    return samples


def draw_column(axes_col, sample, fwd, anchors, show_ctx=128):
    H = sample["future"].size
    fut_x = np.arange(0, H)
    pad_tag = " [pad]" if sample.get("padded") else ""
    label = (f"{sample['subset'].split('/')[-1][:24]}\n"
             f"({sample['freq']}, series #{sample['series_idx']}, "
             f"real_ctx={sample['real_ctx_len']}{pad_tag})")

    real_tail = min(show_ctx, sample["real_ctx_len"])
    ctx_x = np.arange(-real_tail, 0)
    ax = axes_col[0]
    ax.plot(ctx_x, sample["ctx"][-real_tail:], color="gray", linewidth=1.0, label="context")
    ax.plot(fut_x, sample["future"], color="green", linestyle="--", linewidth=1.6, label="GT")
    ax.plot(fut_x, fwd["pred"], color="black", linewidth=1.3, label="Prism total")
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_title(label, fontsize=7.5)
    ax.legend(fontsize=6.5, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes_col[1]
    ax.plot(fut_x, sample["future"], color="green", linestyle="--", linewidth=1.0, alpha=0.4, label="GT")
    ax.plot(fut_x, fwd["trend"], color="#e74c3c", linewidth=1.7,
            label=f"trend ({anchors['t']} anchors→linear)")
    # Anchor positions for F.interpolate(mode='linear', align_corners=True):
    # first/last anchor at endpoints 0 and H-1, evenly spaced.
    ax_an = np.linspace(0, H - 1, anchors["t"])
    ax.plot(ax_an, np.interp(ax_an, fut_x, fwd["trend"]), "o", color="#c0392b", markersize=4)
    ax.set_title("trend", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    ax = axes_col[2]
    ax.plot(fut_x, fwd["s_lo"], color="#1f77b4", linewidth=1.0, alpha=0.55,
            label=f"lo ({anchors['s_lo']})")
    ax.plot(fut_x, fwd["s_mid"], color="#2ca02c", linewidth=1.0, alpha=0.55,
            label=f"mid ({anchors['s_mid']})")
    ax.plot(fut_x, fwd["s_hi"], color="#9467bd", linewidth=1.0, alpha=0.55,
            label=f"hi ({anchors['s_hi']})")
    ax.plot(fut_x, fwd["s_total"], color="black", linewidth=1.4, label="seasonal sum")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_title("seasonal (lo/mid/hi)", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    ax = axes_col[3]
    ax.plot(fut_x, fwd["residual"], color="#2ecc71", linewidth=1.2, label="residual (full 128)")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_title("residual", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Loading Prism from {args.ckpt} on {device} ...")
    model, cfg = load_prism(args.ckpt, device)
    ctx_len = cfg["context_len"]
    horizon = cfg["horizon"]
    anchors = {
        "t": max(1, horizon // cfg["n_freq_downsample_trend"]),
        "s_lo": max(1, horizon // cfg["n_freq_downsample_seasonal"][0]),
        "s_mid": max(1, horizon // cfg["n_freq_downsample_seasonal"][1]),
        "s_hi": max(1, horizon // cfg["n_freq_downsample_seasonal"][2]),
    }
    print(f"  ctx={ctx_len} horizon={horizon} anchors={anchors}")

    rng = np.random.default_rng(args.seed)
    all_samples = []
    for base, _ in SUBSETS:
        print(f"Sampling from monash/{base} ...")
        all_samples.extend(sample_windows(base, args.per_subset, ctx_len, horizon,
                                          cfg["patch_len"], rng))

    print(f"Running {len(all_samples)} forward passes on {device} ...")
    forwards = []
    for i, s in enumerate(all_samples):
        fwd = prism_forward(model, s["ctx"], s["mask"], device)
        forwards.append(fwd)
        print(f"  [{i + 1}/{len(all_samples)}] {s['subset']} #{s['series_idx']} done")

    n = len(all_samples)
    fig, axes = plt.subplots(4, n, figsize=(2.7 * n, 11))
    if n == 1:
        axes = axes.reshape(4, 1)
    for col, (s, fwd) in enumerate(zip(all_samples, forwards)):
        draw_column(axes[:, col], s, fwd, anchors)
    fig.suptitle(
        f"Prism decomposition on training-distribution windows "
        f"(trend={anchors['t']} anchors / seasonal lo={anchors['s_lo']}, mid={anchors['s_mid']}, hi={anchors['s_hi']} cubic)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
