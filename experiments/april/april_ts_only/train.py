"""April-TS-Only: Trend + Seasonal 디코더만 사용, Residual 없음.

핵심:
  - Stage 1: trend 디코더 학습 (타겟 = future)
  - Stage 2: seasonal 디코더 학습 (타겟 = future - trend)
  - 예측 = trend + seasonal (residual 디코더 없음)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
APRIL_DIR = os.path.join(PROJECT_ROOT, "experiments", "april")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APRIL_DIR)

from _common.nhits_baseline import train_and_eval_nhits
from april.model.decomp_timesfm import DecompTimesFM
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

TRAIN_BORDER = 8640
VAL_BORDER = 8640 + 2880

DEFAULT_CONFIG = {
    "n_freq_downsample": [24, 8, 1],
    "interp_trend": "linear",
    "interp_seasonal": "cubic",
    "mlp_units": [[512, 512], [512, 512], [512, 512]],
    "activation": "ReLU",
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "max_steps_per_stage": 5000,
    "batch_size": 32,
    "context_len": 512,
    "embed_dim": 1280,
    "patch_len": 32,
}


def get_config():
    parser = argparse.ArgumentParser(description="April-TS-Only")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--max_steps_per_stage", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--n_freq_downsample", type=str, default="24,8,1",
        help="Frequency downsample ratios for [trend, seasonal, residual]",
    )
    parsed = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg["horizon"] = parsed.horizon
    cfg["max_steps_per_stage"] = parsed.max_steps_per_stage
    cfg["learning_rate"] = parsed.learning_rate
    cfg["batch_size"] = parsed.batch_size
    cfg["loss"] = parsed.loss
    cfg["device"] = parsed.device
    cfg["n_freq_downsample"] = [int(x) for x in parsed.n_freq_downsample.split(",")]
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
        context = torch.tensor(self.target[idx:ctx_end])
        future = torch.tensor(self.target[ctx_end:fut_end])
        masks = torch.zeros(self.context_len, dtype=torch.bool)
        return context, masks, future


def load_ett(name="ETTh1"):
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    return pd.read_csv(url)


# ---------------------------------------------------------------------------
# Training (Stage 1, 2 only)
# ---------------------------------------------------------------------------

def _loss_fn(cfg):
    if cfg["loss"] == "L1":
        return torch.nn.functional.l1_loss
    return torch.nn.functional.mse_loss


def train_stage(model, stage, train_loader, cfg, device):
    assert stage in (1, 2), "TS-Only: stage 1 (trend) or 2 (seasonal) only"
    max_steps = cfg["max_steps_per_stage"]
    lr = cfg["learning_rate"]
    loss_fn = _loss_fn(cfg)

    params = model.get_stage_params(stage)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    model.train()
    for p in model.decoder_t.parameters():
        p.requires_grad = (stage == 1)
    for p in model.decoder_s.parameters():
        p.requires_grad = (stage == 2)
    for p in model.decoder_r.parameters():
        p.requires_grad = False

    losses = []
    step = 0
    while step < max_steps:
        for context, masks, future in train_loader:
            if step >= max_steps:
                break
            context = context.to(device)
            masks = masks.to(device)
            future = future.to(device)

            emb, mu, sigma = model._encode(context, masks)
            future_n = model._normalize_future(future, mu, sigma)

            if stage == 1:
                wave_t_n = model.decoder_t(emb)
                loss = loss_fn(wave_t_n, future_n)
            elif stage == 2:
                with torch.no_grad():
                    wave_t_n = model.decoder_t(emb)
                wave_s_n = model.decoder_s(emb)
                target = future_n - wave_t_n
                loss = loss_fn(wave_s_n, target.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.get_stage_params(stage), max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 200 == 0 or step == 1:
                print(f"    Stage {stage} | Step {step}/{max_steps} | Loss: {loss.item():.6f}")

    return losses


# ---------------------------------------------------------------------------
# Evaluation (trend + seasonal only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    all_preds, all_futures, all_decomps = [], [], []
    for context, masks, future in loader:
        context, masks, future = context.to(device), masks.to(device), future.to(device)

        emb, mu, sigma = model._encode(context, masks)
        wave_t_n = model.decoder_t(emb)
        wave_s_n = model.decoder_s(emb)

        pred_normed = wave_t_n + wave_s_n
        pred = (pred_normed * sigma + mu).squeeze(-1)

        wave_t = (wave_t_n * sigma + mu).squeeze(-1)
        wave_s = (wave_s_n * sigma).squeeze(-1)

        total_mse += ((pred - future) ** 2).sum().item()
        total_mae += (pred - future).abs().sum().item()
        n += future.numel()
        all_preds.append(pred.cpu())
        all_futures.append(future.cpu())
        all_decomps.append({"trend": wave_t.cpu(), "seasonal": wave_s.cpu()})

    preds = torch.cat(all_preds)
    futures = torch.cat(all_futures)
    decomps = {k: torch.cat([d[k] for d in all_decomps]) for k in all_decomps[0]}
    return total_mse / n, total_mae / n, preds, futures, decomps


def evaluate_baseline(baseline_model, test_ds, horizon, context_len):
    contexts_np, futures_np = [], []
    for i in range(len(test_ds)):
        ctx, _, fut = test_ds[i]
        contexts_np.append(ctx.numpy())
        futures_np.append(fut.numpy())
    all_preds = []
    for start in range(0, len(contexts_np), 64):
        pf, _ = baseline_model.forecast(horizon, contexts_np[start:start + 64])
        all_preds.append(pf)
    all_preds = np.concatenate(all_preds, axis=0)
    all_futures = np.array(futures_np)
    mse = np.mean((all_preds - all_futures) ** 2)
    mae = np.mean(np.abs(all_preds - all_futures))
    return mse, mae, torch.tensor(all_preds, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_stage_losses(all_stage_losses, horizon):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = ["Trend", "Seasonal"]
    colors = ["#e74c3c", "#3498db"]
    for ax, losses, name, color in zip(axes, all_stage_losses, names, colors):
        ax.plot(losses, color=color, alpha=0.6, linewidth=0.8)
        if len(losses) > 20:
            w = min(50, len(losses) // 5)
            ma = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(losses)), ma, color=color, linewidth=2)
        ax.set_title(f"Stage: {name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"April-TS-Only Training Losses (h={horizon})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"training_losses_h{horizon}.png"), dpi=150)
    plt.close()


def plot_decomposition(futures, horizon, april_preds, april_decomps,
                       nhits_preds, nhits_decomps, n_samples=4):
    n = min(n_samples, futures.shape[0])
    ids = np.random.RandomState(42).choice(futures.shape[0], n, replace=False)
    t = np.arange(horizon)

    fig, axes = plt.subplots(n, 2, figsize=(24, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, idx in enumerate(ids):
        gt = futures[idx].numpy()

        ax = axes[row, 0]
        ax.plot(t, gt, label="GT", color="green", linestyle="--", linewidth=2)
        ax.plot(t, april_preds[idx].numpy(), label="Pred (T+S)", color="black", linewidth=1.5)
        ax.plot(t, april_decomps["trend"][idx].numpy(), label="Trend", color="#e74c3c", linewidth=1.2)
        ax.plot(t, april_decomps["seasonal"][idx].numpy(), label="Seasonal", color="#3498db", linewidth=1.2)
        ax.set_title(f"April TS-Only — Sample {idx}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.plot(t, gt, label="GT", color="green", linestyle="--", linewidth=2)
        ax.plot(t, nhits_preds[idx].numpy(), label="Pred", color="black", linewidth=1.5)
        ax.plot(t, nhits_decomps["trend"][idx].numpy(), label="Trend", color="#e74c3c", linewidth=1.2)
        ax.plot(t, nhits_decomps["seasonal"][idx].numpy(), label="Seasonal", color="#3498db", linewidth=1.2)
        ax.plot(t, nhits_decomps["detail"][idx].numpy(), label="Detail", color="#2ecc71", linewidth=1.2, alpha=0.6)
        ax.set_title(f"N-HiTS — Sample {idx}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Decomposition (h={horizon}): April TS-Only vs N-HiTS", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"decomposition_h{horizon}.png"), dpi=150)
    plt.close()


def plot_comparison(futures, bl_preds, april_preds, nhits_preds, horizon):
    n = min(4, futures.shape[0])
    ids = np.random.RandomState(42).choice(futures.shape[0], n, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, idx in zip(axes.flat, ids):
        ax.plot(futures[idx].numpy(), label="GT", color="green", linestyle="--", linewidth=2)
        ax.plot(bl_preds[idx].numpy(), label="TimesFM (AR)", color="blue", alpha=0.7)
        ax.plot(nhits_preds[idx].numpy(), label="N-HiTS", color="#DAA520", linewidth=1.5)
        ax.plot(april_preds[idx].numpy(), label="April TS-Only", color="red", linewidth=1.5)
        mse_a = ((april_preds[idx] - futures[idx]) ** 2).mean().item()
        mse_n = ((nhits_preds[idx] - futures[idx]) ** 2).mean().item()
        ax.set_title(f"Sample {idx} (TS={mse_a:.2f}, NHiTS={mse_n:.2f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"ETTh1 (h={horizon}) — Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_h{horizon}.png"), dpi=150)
    plt.close()


def plot_summary(results):
    horizons = [r["horizon"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [(ax1, "mse", "MSE"), (ax2, "mae", "MAE")]:
        ax.plot(horizons, [r[f"bl_{metric}"] for r in results], "s--", color="blue",
                linewidth=2, markersize=8, label="TimesFM (AR)")
        ax.plot(horizons, [r[f"nhits_{metric}"] for r in results], "D--", color="#DAA520",
                linewidth=2, markersize=8, label="N-HiTS")
        ax.plot(horizons, [r[f"test_{metric}"] for r in results], "o-", color="red",
                linewidth=2, markersize=8, label="April TS-Only")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(title)
        ax.set_title(f"ETTh1 Test {title}")
        ax.set_xticks(horizons)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("April TS-Only — Horizon Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "horizon_comparison.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_horizon(cfg, df, device, baseline_model):
    horizon = cfg["horizon"]
    context_len = cfg["context_len"]
    batch_size = cfg["batch_size"]

    print(f"\n{'='*60}")
    print(f"  Horizon = {horizon}")
    print(f"{'='*60}")

    train_ds = ETTDataset(df.iloc[:TRAIN_BORDER], context_len, horizon)
    test_ds = ETTDataset(df.iloc[VAL_BORDER:], context_len, horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # === April TS-Only ===
    print("\n[April TS-Only] 모델 로딩 중...")
    model = DecompTimesFM(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    freq_ds = cfg["n_freq_downsample"]
    print(f"계수 수: trend={horizon//freq_ds[0]}, seasonal={horizon//freq_ds[1]} (residual 미사용)")

    all_stage_losses = []
    for stage in [1, 2]:
        names = {1: "Trend", 2: "Seasonal"}
        print(f"\n--- Stage {stage}: {names[stage]} ---")
        losses = train_stage(model, stage, train_loader, cfg, device)
        all_stage_losses.append(losses)

    plot_stage_losses(all_stage_losses, horizon)

    torch.save({
        "model_state_dict": {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("backbone.")
        },
        "config": cfg,
    }, os.path.join(OUTPUT_DIR, f"april_ts_only_h{horizon}.pt"))

    # === N-HiTS (캐싱) ===
    cache_dir = os.path.join(EXPERIMENTS_DIR, "bench_cache")
    os.makedirs(cache_dir, exist_ok=True)
    nhits_mse, nhits_mae, nhits_preds, nhits_decomps = train_and_eval_nhits(
        df=df, target_col="OT",
        train_border=TRAIN_BORDER, val_border=VAL_BORDER,
        context_len=context_len, horizon=horizon,
        max_steps=cfg["max_steps_per_stage"] * 3,
        cache_path=os.path.join(cache_dir, f"nhits_orig_cache_h{horizon}.pt"),
    )

    # === Evaluation ===
    print("\n--- Evaluation ---")
    test_mse, test_mae, test_preds, test_futures, test_decomps = evaluate(
        model, test_loader, device,
    )

    bl_cache = os.path.join(cache_dir, f"baseline_cache_h{horizon}.pt")
    if os.path.exists(bl_cache):
        bc = torch.load(bl_cache, map_location="cpu", weights_only=False)
        bl_mse, bl_mae, bl_preds = bc["mse"], bc["mae"], bc["preds"]
    else:
        max_h = ((max(horizon, 128) - 1) // 128 + 1) * 128
        baseline_model.compile(ForecastConfig(
            max_context=context_len, max_horizon=max_h,
            per_core_batch_size=64, force_flip_invariance=True, infer_is_positive=False,
        ))
        bl_mse, bl_mae, bl_preds = evaluate_baseline(baseline_model, test_ds, horizon, context_len)
        torch.save({"mse": bl_mse, "mae": bl_mae, "preds": bl_preds}, bl_cache)

    print(f"\n  ETTh1 Test (h={horizon}):")
    print(f"    TimesFM (AR)    | MSE: {bl_mse:.4f} | MAE: {bl_mae:.4f}")
    print(f"    N-HiTS          | MSE: {nhits_mse:.4f} | MAE: {nhits_mae:.4f}")
    print(f"    April TS-Only   | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

    plot_decomposition(test_futures, horizon, test_preds, test_decomps,
                       nhits_preds, nhits_decomps)
    plot_comparison(test_futures, bl_preds, test_preds, nhits_preds, horizon)

    return {
        "horizon": horizon,
        "test_mse": float(test_mse), "test_mae": float(test_mae),
        "nhits_mse": float(nhits_mse), "nhits_mae": float(nhits_mae),
        "bl_mse": float(bl_mse), "bl_mae": float(bl_mae),
    }


def main():
    cfg = get_config()
    device = torch.device(cfg["device"])

    print(f"Device: {device}")
    print(f"Loss: {cfg['loss']}")
    print(f"n_freq_downsample: {cfg['n_freq_downsample']}")
    print(f"Output: {OUTPUT_DIR}")

    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    df = load_ett("ETTh1")
    baseline_model = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False,
    )

    results = []
    for h in [96, 192, 336, 720]:
        cfg["horizon"] = h
        results.append(run_single_horizon(cfg, df, device, baseline_model))

    print(f"\n{'='*60}")
    print("  Final Results (ETTh1 Test)")
    print(f"{'='*60}")
    print(f"  {'H':>6s}  {'AR MSE':>10s}  {'NHiTS MSE':>10s}  {'TS MSE':>10s}  {'TS MAE':>10s}")
    for r in results:
        print(f"  {r['horizon']:>6d}  {r['bl_mse']:>10.4f}  {r['nhits_mse']:>10.4f}  {r['test_mse']:>10.4f}  {r['test_mae']:>10.4f}")

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_summary(results)

    from bench_utils import update_benchmark
    update_benchmark("April-TS-Only", results, steps_per_stage=cfg["max_steps_per_stage"])
    print(f"\n모든 결과물이 {OUTPUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
