"""April L1: AvgPool 타겟 분해 + MAE loss."""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
APRIL_DIR = os.path.join(PROJECT_ROOT, "experiments", "april")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APRIL_DIR)

from _common.eval_decomposition import (
    compute_decomposition_metrics,
    print_decomposition_metrics,
    plot_decomposition_metrics,
)
from _common.nhits_baseline import train_and_eval_nhits
from april.model.decomp_timesfm import DecompTimesFM
from april.model.decoder_t import make_trend_target
from april.model.decoder_s import make_seasonal_target
from april.config import DEFAULT_CONFIG
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")  # for bench_cache/ (shared)

TRAIN_BORDER = 8640
VAL_BORDER = 8640 + 2880
LOSS_FN = F.l1_loss
LOSS_NAME = "L1"

# AvgPool 타겟 생성용 커널
TARGET_POOL_KERNELS = [8, 4, 1]


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--max_steps_per_stage", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parsed = parser.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["horizon"] = parsed.horizon
    cfg["max_steps_per_stage"] = parsed.max_steps_per_stage
    cfg["learning_rate"] = parsed.learning_rate
    cfg["batch_size"] = parsed.batch_size
    cfg["device"] = parsed.device
    cfg["loss"] = LOSS_NAME
    return cfg


class ETTDataset(Dataset):
    def __init__(self, df, context_len, horizon, target_col="OT"):
        self.context_len = context_len
        self.horizon = horizon
        self.target = df[target_col].values.astype(np.float32)
        self.n_samples = len(self.target) - context_len - horizon + 1
    def __len__(self):
        return max(0, self.n_samples)
    def __getitem__(self, idx):
        ctx_end = idx + self.context_len
        fut_end = ctx_end + self.horizon
        return (torch.tensor(self.target[idx:ctx_end]),
                torch.zeros(self.context_len, dtype=torch.bool),
                torch.tensor(self.target[ctx_end:fut_end]))


def load_ett(name="ETTh1"):
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    return pd.read_csv(url)


def train_stage(model, stage, train_loader, cfg, device):
    max_steps = cfg["max_steps_per_stage"]
    params = model.get_stage_params(stage)
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    model.train()
    for p in model.decoder_t.parameters(): p.requires_grad = (stage == 1)
    for p in model.decoder_s.parameters(): p.requires_grad = (stage == 2)
    for p in model.decoder_r.parameters(): p.requires_grad = (stage == 3)

    losses = []
    step = 0
    while step < max_steps:
        for context, masks, future in train_loader:
            if step >= max_steps: break
            context, masks, future = context.to(device), masks.to(device), future.to(device)

            if stage == 1:
                wave_t_n, mu, sigma = model.forward_stage1(context, masks)
                future_n = model._normalize_future(future, mu, sigma)
                target = make_trend_target(future_n, TARGET_POOL_KERNELS[0])
                loss = LOSS_FN(wave_t_n, target)
            elif stage == 2:
                wave_t_n, wave_s_n, mu, sigma = model.forward_stage2(context, masks)
                future_n = model._normalize_future(future, mu, sigma)
                r1_n = future_n - wave_t_n.detach()
                target = make_seasonal_target(r1_n, TARGET_POOL_KERNELS[1])
                loss = LOSS_FN(wave_s_n, target)
            elif stage == 3:
                emb, mu, sigma = model._encode(context, masks)
                future_n = model._normalize_future(future, mu, sigma)
                with torch.no_grad():
                    wt_n = model.decoder_t(emb)
                    ws_n = model.decoder_s(emb)
                    r2_n = future_n - wt_n - ws_n
                residual_n = model.decoder_r(emb)
                loss = LOSS_FN(residual_n, r2_n.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_stage_params(stage), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0 or step == 1:
                print(f"    Stage {stage} | Step {step}/{max_steps} | Loss: {loss.item():.6f}")
    return losses


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse, total_mae, n = 0., 0., 0
    all_p, all_f, all_d = [], [], []
    for ctx, msk, fut in loader:
        ctx, msk, fut = ctx.to(device), msk.to(device), fut.to(device)
        pred, decomp = model(ctx, msk)
        total_mse += ((pred - fut)**2).sum().item()
        total_mae += (pred - fut).abs().sum().item()
        n += fut.numel()
        all_p.append(pred.cpu()); all_f.append(fut.cpu())
        all_d.append({k: v.cpu() for k, v in decomp.items()})
    p, f = torch.cat(all_p), torch.cat(all_f)
    d = {k: torch.cat([x[k] for x in all_d]) for k in all_d[0]}
    return total_mse/n, total_mae/n, p, f, d


def evaluate_baseline(bl, test_ds, horizon, context_len):
    ctxs, futs = [], []
    for i in range(len(test_ds)):
        c, _, f = test_ds[i]; ctxs.append(c.numpy()); futs.append(f.numpy())
    preds = []
    for s in range(0, len(ctxs), 64):
        pf, _ = bl.forecast(horizon, ctxs[s:s+64]); preds.append(pf)
    preds = np.concatenate(preds); futs = np.array(futs)
    return float(np.mean((preds-futs)**2)), float(np.mean(np.abs(preds-futs))), torch.tensor(preds, dtype=torch.float32)


# --- Plots ---

def plot_stage_losses(losses_list, horizon):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, losses, name, c in zip(axes, losses_list,
            ["Trend","Seasonal","Residual"], ["#e74c3c","#3498db","#2ecc71"]):
        ax.plot(losses, color=c, alpha=0.6, linewidth=0.8)
        if len(losses) > 20:
            w = min(50, len(losses)//5)
            ma = np.convolve(losses, np.ones(w)/w, mode="valid")
            ax.plot(range(w-1, len(losses)), ma, color=c, linewidth=2)
        ax.set_title(f"Stage: {name}"); ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    plt.suptitle(f"April {LOSS_NAME} Training Losses (h={horizon})"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"training_losses_h{horizon}.png"), dpi=150); plt.close()

def plot_decomposition(futures, horizon, ap, ad, np_, nd, n_samples=4):
    n = min(n_samples, futures.shape[0])
    ids = np.random.RandomState(42).choice(futures.shape[0], n, replace=False)
    t = np.arange(horizon)
    fig, axes = plt.subplots(n, 2, figsize=(24, 4*n))
    if n == 1: axes = axes.reshape(1, 2)
    for row, idx in enumerate(ids):
        gt = futures[idx].numpy()
        for col, (preds, decomps, title) in enumerate([
            (ap, ad, f"April {LOSS_NAME}"), (np_, nd, "N-HiTS")]):
            ax = axes[row, col]
            ax.plot(t, gt, label="GT", color="green", linestyle="--", linewidth=2)
            ax.plot(t, preds[idx].numpy(), label="Pred", color="black", linewidth=1.5)
            dk = list(decomps.keys())
            for k, c in zip(dk, ["#e74c3c","#3498db","#2ecc71"]):
                ax.plot(t, decomps[k][idx].numpy(), label=k.capitalize(), color=c, linewidth=1.2)
            ax.set_title(f"{title} — Sample {idx}"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Decomposition (h={horizon}): April {LOSS_NAME} vs N-HiTS", fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f"decomposition_h{horizon}.png"), dpi=150); plt.close()

def plot_comparison(futures, bl_p, ap, np_, horizon):
    n = min(4, futures.shape[0])
    ids = np.random.RandomState(42).choice(futures.shape[0], n, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, idx in zip(axes.flat, ids):
        ax.plot(futures[idx].numpy(), label="GT", color="green", linestyle="--", linewidth=2)
        ax.plot(bl_p[idx].numpy(), label="TimesFM", color="blue", alpha=0.7)
        ax.plot(np_[idx].numpy(), label="N-HiTS", color="#DAA520", linewidth=1.5)
        ax.plot(ap[idx].numpy(), label=f"April {LOSS_NAME}", color="red", linewidth=1.5)
        ma = ((ap[idx]-futures[idx])**2).mean().item()
        mn = ((np_[idx]-futures[idx])**2).mean().item()
        ax.set_title(f"Sample {idx} (April={ma:.2f}, NHiTS={mn:.2f})")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle(f"ETTh1 (h={horizon}) — Comparison"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_h{horizon}.png"), dpi=150); plt.close()

def plot_summary(results):
    horizons = [r["horizon"] for r in results]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m, t in [(a1,"mse","MSE"),(a2,"mae","MAE")]:
        ax.plot(horizons, [r[f"bl_{m}"] for r in results], "s--", color="blue", linewidth=2, markersize=8, label="TimesFM")
        ax.plot(horizons, [r[f"nhits_{m}"] for r in results], "D--", color="#DAA520", linewidth=2, markersize=8, label="N-HiTS")
        ax.plot(horizons, [r[f"test_{m}"] for r in results], "o-", color="red", linewidth=2, markersize=8, label=f"April {LOSS_NAME}")
        ax.set_xlabel("Horizon"); ax.set_ylabel(t); ax.set_title(f"ETTh1 Test {t}")
        ax.set_xticks(horizons); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Horizon Comparison — April {LOSS_NAME}"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "horizon_comparison.png"), dpi=150); plt.close()


# --- Main ---

def run_single_horizon(cfg, df, device, baseline_model):
    horizon = cfg["horizon"]; context_len = cfg["context_len"]; bs = cfg["batch_size"]
    print(f"\n{'='*60}\n  Horizon = {horizon}\n{'='*60}")

    train_ds = ETTDataset(df.iloc[:TRAIN_BORDER], context_len, horizon)
    test_ds = ETTDataset(df.iloc[VAL_BORDER:], context_len, horizon)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # April
    print(f"\n[April {LOSS_NAME}] 모델 로딩 중...")
    model = DecompTimesFM(cfg).to(device)
    losses_all = []
    for stage in [1, 2, 3]:
        print(f"\n--- Stage {stage}: {['','Trend','Seasonal','Residual'][stage]} ---")
        losses_all.append(train_stage(model, stage, train_loader, cfg, device))
    plot_stage_losses(losses_all, horizon)

    # N-HiTS baseline (원본 neuralforecast 구현, cached)
    cache_dir = os.path.join(EXPERIMENTS_DIR, "bench_cache"); os.makedirs(cache_dir, exist_ok=True)
    nhits_mse, nhits_mae, nhits_preds, nhits_decomps = train_and_eval_nhits(
        df=df, target_col="OT",
        train_border=TRAIN_BORDER, val_border=VAL_BORDER,
        context_len=context_len, horizon=horizon,
        max_steps=cfg["max_steps_per_stage"] * 3,
        cache_path=os.path.join(cache_dir, f"nhits_orig_cache_h{horizon}.pt"),
    )

    # Eval
    test_mse, test_mae, test_preds, test_futures, test_decomps = evaluate(model, test_loader, device)

    bl_path = os.path.join(cache_dir, f"baseline_cache_h{horizon}.pt")
    if os.path.exists(bl_path):
        bc = torch.load(bl_path, map_location="cpu", weights_only=False)
        bl_mse, bl_mae, bl_preds = bc["mse"], bc["mae"], bc["preds"]
    else:
        max_h = ((max(horizon,128)-1)//128+1)*128
        baseline_model.compile(ForecastConfig(max_context=context_len, max_horizon=max_h, per_core_batch_size=64, force_flip_invariance=True, infer_is_positive=False))
        bl_mse, bl_mae, bl_preds = evaluate_baseline(baseline_model, test_ds, horizon, context_len)
        torch.save({"mse":bl_mse,"mae":bl_mae,"preds":bl_preds}, bl_path)

    print(f"\n  ETTh1 Test (h={horizon}):")
    print(f"    TimesFM        | MSE: {bl_mse:.4f} | MAE: {bl_mae:.4f}")
    print(f"    N-HiTS         | MSE: {nhits_mse:.4f} | MAE: {nhits_mae:.4f}")
    print(f"    April {LOSS_NAME}      | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

    plot_decomposition(test_futures, horizon, test_preds, test_decomps, nhits_preds, nhits_decomps)
    plot_comparison(test_futures, bl_preds, test_preds, nhits_preds, horizon)

    # --- Decomposition quality metrics (April) ---
    decomp_metrics = compute_decomposition_metrics(
        test_futures, test_decomps, period=24,
    )
    print_decomposition_metrics(decomp_metrics, horizon, label=f"April {LOSS_NAME}")
    plot_decomposition_metrics(
        decomp_metrics, horizon,
        os.path.join(OUTPUT_DIR, f"decomposition_metrics_h{horizon}.png"),
        title_prefix=f"April {LOSS_NAME} — ",
    )

    # --- Decomposition quality metrics (N-HiTS, for comparison) ---
    nhits_decomps_renamed = {
        "trend":    nhits_decomps["trend"],
        "seasonal": nhits_decomps["seasonal"],
        "residual": nhits_decomps["detail"],
    }
    nhits_decomp_metrics = compute_decomposition_metrics(
        test_futures, nhits_decomps_renamed, period=24,
    )
    print_decomposition_metrics(nhits_decomp_metrics, horizon, label="N-HiTS")
    plot_decomposition_metrics(
        nhits_decomp_metrics, horizon,
        os.path.join(OUTPUT_DIR, f"decomposition_metrics_nhits_h{horizon}.png"),
        title_prefix="N-HiTS — ",
    )

    return {"horizon":horizon, "test_mse":float(test_mse), "test_mae":float(test_mae),
            "nhits_mse":float(nhits_mse), "nhits_mae":float(nhits_mae),
            "bl_mse":float(bl_mse), "bl_mae":float(bl_mae),
            "decomposition_metrics": decomp_metrics,
            "nhits_decomposition_metrics": nhits_decomp_metrics}


def main():
    cfg = get_config(); device = torch.device(cfg["device"])
    print(f"Device: {device}\nLoss: {LOSS_NAME}\nOutput: {OUTPUT_DIR}")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f: json.dump(cfg, f, indent=2)
    df = load_ett("ETTh1")
    bl = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    results = []
    for h in [96, 192, 336, 720]:
        cfg["horizon"] = h; results.append(run_single_horizon(cfg, df, device, bl))
    print(f"\n{'='*60}\n  Final Results (ETTh1 Test)\n{'='*60}")
    for r in results:
        print(f"  h={r['horizon']:>4d}  AR={r['bl_mse']:.4f}  NHiTS={r['nhits_mse']:.4f}  April={r['test_mse']:.4f}")
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f: json.dump(results, f, indent=2)
    plot_summary(results)
    from bench_utils import update_benchmark
    update_benchmark("April L1", results, steps_per_stage=cfg["max_steps_per_stage"])

if __name__ == "__main__":
    main()
