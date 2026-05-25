"""Evaluate prism on the 100 synthetic series — mirror of eval_april_fixed.py.

Prism is the multi-frequency variant (3 seasonal decoders: lo/mid/hi). For
parity with april_fixed we report on the aggregated trend/seasonal/residual
triple, but also save the per-band seasonal arrays for diagnostics.

Outputs (eval/artificial/):
  results_prism.json       — per-horizon MAE/MSE + decomp metrics
  predictions_prism.npz    — preds[h] arrays + decomp arrays
  forecast_samples_prism.png
  decomp_samples_prism.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
APRIL_EXP_DIR = HERE.parents[1]
PROJECT_ROOT = APRIL_EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APRIL_EXP_DIR))

from _common.ett_eval import forecast_native_ar  # noqa: E402

DEVICE = torch.device("cpu")
CKPT = APRIL_EXP_DIR / "pretrain" / "prism" / "prism_pretrain_final.pt"
CTX_LEN = 1024
HORIZONS = [96, 192, 336, 720]
BATCH = 4
N_BEST = 3
N_RANDOM = 2
SEED = 1


def load_model():
    from prism.model.decomp_timesfm_prism import DecompTimesFMPrism
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = state["cfg"]
    model = DecompTimesFMPrism(cfg).to(DEVICE)
    model.load_state_dict(state["decoder_state_dict"], strict=False)
    model.eval()
    print(f"loaded {CKPT.name}: ctx={cfg['context_len']}, horizon={cfg['horizon']}")
    return model, cfg


def forecast_metrics(model, series: np.ndarray) -> tuple[dict, dict]:
    N, T = series.shape
    ctxs = series[:, :CTX_LEN].astype(np.float32)
    metrics = {}
    preds_by_h = {}
    for h in HORIZONS:
        if CTX_LEN + h > T:
            continue
        gt = series[:, CTX_LEN : CTX_LEN + h]
        std = np.clip(series.std(axis=1, keepdims=True), 1e-6, None)
        t0 = time.time()
        print(f"  h={h}: forecasting {N} series with batch={BATCH}...")
        preds = forecast_native_ar(model, ctxs, h, batch_size=BATCH,
                                   device=DEVICE, flip_invariance=True)
        elapsed = time.time() - t0
        err = preds - gt
        mae_raw = float(np.mean(np.abs(err)))
        mse_raw = float(np.mean(err * err))
        mae_norm = float(np.mean(np.abs(err) / std))
        mse_norm = float(np.mean((err * err) / (std * std)))
        metrics[str(h)] = {
            "MAE_raw": mae_raw, "MSE_raw": mse_raw,
            "MAE_norm": mae_norm, "MSE_norm": mse_norm,
            "elapsed_sec": elapsed,
        }
        preds_by_h[str(h)] = preds.astype(np.float32)
        print(f"    MAE_raw={mae_raw:.4f}  MSE_raw={mse_raw:.4f}  "
              f"MAE_norm={mae_norm:.4f}  MSE_norm={mse_norm:.4f}  "
              f"({elapsed:.1f}s)")
    return metrics, preds_by_h


@torch.no_grad()
def decomp_metrics(model, series: np.ndarray, components: np.ndarray) -> tuple[dict, dict]:
    N, T = series.shape
    h = int(model.horizon)
    ctxs = torch.from_numpy(series[:, :CTX_LEN].astype(np.float32))
    gt_trend = components[:, 0, CTX_LEN : CTX_LEN + h].astype(np.float32)
    gt_seasonal = components[:, 1, CTX_LEN : CTX_LEN + h].astype(np.float32)
    gt_residual = components[:, 2, CTX_LEN : CTX_LEN + h].astype(np.float32)

    pred_trend = np.empty((N, h), dtype=np.float32)
    pred_seasonal = np.empty((N, h), dtype=np.float32)
    pred_seasonal_lo = np.empty((N, h), dtype=np.float32)
    pred_seasonal_mid = np.empty((N, h), dtype=np.float32)
    pred_seasonal_hi = np.empty((N, h), dtype=np.float32)
    pred_residual = np.empty((N, h), dtype=np.float32)
    pred_sum = np.empty((N, h), dtype=np.float32)
    print(f"  decomp h={h} batch={BATCH}...")
    t0 = time.time()
    for i in range(0, N, BATCH):
        ctx = ctxs[i : i + BATCH].to(DEVICE)
        msk = torch.zeros_like(ctx, dtype=torch.bool)
        pred, decomp = model(ctx, msk)
        b = ctx.shape[0]
        pred_sum[i : i + b] = pred.cpu().numpy()
        pred_trend[i : i + b] = decomp["trend"].cpu().numpy()
        pred_seasonal[i : i + b] = decomp["seasonal"].cpu().numpy()
        pred_seasonal_lo[i : i + b] = decomp["seasonal_lo"].cpu().numpy()
        pred_seasonal_mid[i : i + b] = decomp["seasonal_mid"].cpu().numpy()
        pred_seasonal_hi[i : i + b] = decomp["seasonal_hi"].cpu().numpy()
        pred_residual[i : i + b] = decomp["residual"].cpu().numpy()
    elapsed = time.time() - t0
    print(f"    decomp forward done ({elapsed:.1f}s)")

    def _demean(x):
        return x - x.mean(axis=1, keepdims=True)

    gt_sum = gt_trend + gt_seasonal + gt_residual
    metrics = {
        "horizon": h,
        "combined": _named_errors(pred_sum, gt_sum),
        "trend":    _named_errors(_demean(pred_trend), _demean(gt_trend)),
        "seasonal": _named_errors(pred_seasonal, gt_seasonal),
        "seasonal_lo":  _named_errors(pred_seasonal_lo, gt_seasonal),
        "seasonal_mid": _named_errors(pred_seasonal_mid, gt_seasonal),
        "seasonal_hi":  _named_errors(pred_seasonal_hi, gt_seasonal),
        "residual": _named_errors(pred_residual, gt_residual),
        "elapsed_sec": elapsed,
    }
    arrays = {
        "pred_sum": pred_sum,
        "pred_trend": pred_trend,
        "pred_seasonal": pred_seasonal,
        "pred_seasonal_lo": pred_seasonal_lo,
        "pred_seasonal_mid": pred_seasonal_mid,
        "pred_seasonal_hi": pred_seasonal_hi,
        "pred_residual": pred_residual,
        "gt_trend": gt_trend,
        "gt_seasonal": gt_seasonal,
        "gt_residual": gt_residual,
    }
    for kind in ["trend", "seasonal", "residual", "combined"]:
        m = metrics[kind]
        print(f"    {kind:10s}: MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  "
              f"corr={m['corr']:+.3f}")
    return metrics, arrays


def _named_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    p_c = pred - pred.mean(axis=1, keepdims=True)
    g_c = gt - gt.mean(axis=1, keepdims=True)
    num = (p_c * g_c).sum(axis=1)
    den = np.sqrt((p_c * p_c).sum(axis=1) * (g_c * g_c).sum(axis=1))
    corr = float(np.mean(np.where(den > 1e-8, num / np.clip(den, 1e-8, None), 0.0)))
    return {"MAE": mae, "MSE": mse, "corr": corr}


def select_plot_idx(series, preds_by_h, n_best=N_BEST, n_random=N_RANDOM,
                    seed=SEED):
    h_focus = HORIZONS[-1] if str(HORIZONS[-1]) in preds_by_h else \
        max(int(k) for k in preds_by_h.keys())
    preds = preds_by_h[str(h_focus)]
    gt = series[:, CTX_LEN : CTX_LEN + h_focus]
    mae_per = np.mean(np.abs(preds - gt), axis=1)
    order = np.argsort(mae_per)
    best = order[:n_best].tolist()
    rng = np.random.default_rng(seed)
    rand = rng.choice(order[n_best:], size=n_random, replace=False).tolist()
    sel = sorted(best + rand)
    is_best = [i in best for i in sel]
    print(f"  ranking by h={h_focus} MAE: best 3 idx={best} "
          f"(MAE={mae_per[best].tolist()}), random 2 idx={rand}")
    return np.array(sel), is_best


def plot_forecast(series, preds_by_h, idx, is_best, out_path):
    h_focus = HORIZONS[-1]
    if str(h_focus) not in preds_by_h:
        h_focus = max(int(k) for k in preds_by_h.keys())
    preds = preds_by_h[str(h_focus)]
    n = len(idx)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), squeeze=False)
    show_ctx = 192
    for row, i in enumerate(idx):
        ax = axes[row, 0]
        ctx_tail = series[i, CTX_LEN - show_ctx : CTX_LEN]
        gt = series[i, CTX_LEN : CTX_LEN + h_focus]
        x_ctx = np.arange(-show_ctx, 0)
        x_fwd = np.arange(h_focus)
        mae = float(np.mean(np.abs(preds[i] - gt)))
        tag = "BEST" if is_best[row] else "rand"
        ax.plot(x_ctx, ctx_tail, color="gray", linewidth=0.8, label="ctx (tail)")
        ax.plot(x_fwd, gt, color="green", linewidth=1.2, label="GT")
        ax.plot(x_fwd, preds[i], color="#9467bd", linewidth=1.0,
                label=f"prism pred (MAE={mae:.3f})")
        ax.axvline(0, color="k", linewidth=0.4, alpha=0.4)
        ax.set_title(f"[{tag}] series #{i} — h={h_focus}", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"prism forecast on artificial corpus (h={h_focus}) — "
                 f"best 3 (lowest MAE) + 2 random", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_decomp(arrays, idx, is_best, series, out_path):
    h = arrays["pred_trend"].shape[1]
    n = len(idx)
    fig, axes = plt.subplots(n, 4, figsize=(20, 2.8 * n), squeeze=False)
    x = np.arange(h)
    for row, i in enumerate(idx):
        tag = "BEST" if is_best[row] else "rand"
        for col, name in enumerate(["trend", "seasonal", "residual"]):
            ax = axes[row, col]
            gt = arrays[f"gt_{name}"][i]
            pred = arrays[f"pred_{name}"][i]
            if name == "trend":
                gt = gt - gt.mean()
                pred = pred - pred.mean()
            ax.plot(x, gt, color="green", linewidth=1.1, label="GT")
            ax.plot(x, pred, color="#9467bd", linewidth=0.9, label="pred")
            if name == "seasonal":
                # Overlay multi-band breakdown faintly
                ax.plot(x, arrays["pred_seasonal_lo"][i], color="#1f77b4",
                        linewidth=0.5, alpha=0.55, label="lo")
                ax.plot(x, arrays["pred_seasonal_mid"][i], color="#ff7f0e",
                        linewidth=0.5, alpha=0.55, label="mid")
                ax.plot(x, arrays["pred_seasonal_hi"][i], color="#d62728",
                        linewidth=0.5, alpha=0.55, label="hi")
            ax.set_title(f"[{tag}] #{i} {name}", fontsize=9)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax = axes[row, 3]
        gt_total = series[i, CTX_LEN : CTX_LEN + h]
        pred_total = arrays["pred_sum"][i]
        mae = float(np.mean(np.abs(pred_total - gt_total)))
        ax.plot(x, gt_total, color="green", linewidth=1.1, label="GT")
        ax.plot(x, pred_total, color="#9467bd", linewidth=0.9,
                label=f"pred (MAE={mae:.3f})")
        ax.set_title(f"[{tag}] #{i} combined", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("prism single-chunk (h=128) decomposition vs GT — "
                 "best 3 (lowest h=720 MAE) + 2 random", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print(f"device={DEVICE}")
    series = np.load(HERE / "series.npy")
    components = np.load(HERE / "components.npy")
    print(f"loaded series {series.shape}, components {components.shape}")

    model, cfg = load_model()

    print("\n=== Forecast metrics ===")
    fcst_metrics, preds_by_h = forecast_metrics(model, series)

    print("\n=== Decomposition fidelity (h=128) ===")
    dcmp_metrics, arrays = decomp_metrics(model, series, components)

    plot_idx, is_best = select_plot_idx(series, preds_by_h)

    print("\n=== Plots ===")
    plot_forecast(series, preds_by_h, plot_idx, is_best,
                  HERE / "forecast_samples_prism.png")
    plot_decomp(arrays, plot_idx, is_best, series,
                HERE / "decomp_samples_prism.png")

    np.savez(HERE / "predictions_prism.npz",
             plot_idx=plot_idx,
             **{f"preds_h{k}": v for k, v in preds_by_h.items()},
             **{f"decomp_{k}": v for k, v in arrays.items()})
    print(f"  wrote {HERE / 'predictions_prism.npz'}")

    results = {
        "model": "prism",
        "ckpt": str(CKPT),
        "context_len": CTX_LEN,
        "n_series": int(series.shape[0]),
        "series_len": int(series.shape[1]),
        "device": str(DEVICE),
        "forecast": fcst_metrics,
        "decomposition": dcmp_metrics,
    }
    (HERE / "results_prism.json").write_text(json.dumps(results, indent=2))
    print(f"  wrote {HERE / 'results_prism.json'}")


if __name__ == "__main__":
    main()
