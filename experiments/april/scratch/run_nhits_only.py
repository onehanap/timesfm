"""N-HiTS만 재학습 + 호리즌별 디컴포지션 플롯 (기존 실험과 같은 샘플 사용)."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
APRIL_DIR = os.path.join(PROJECT_ROOT, "experiments", "april")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, APRIL_DIR)

from _common.nhits_baseline import train_and_eval_nhits

TRAIN_BORDER = 8640
VAL_BORDER = 8640 + 2880
CONTEXT_LEN = 512
HORIZONS = [96, 192, 336, 720]
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "experiments", "bench_cache")


def load_ett(name="ETTh1"):
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
    return pd.read_csv(url)


def plot_nhits_decomposition(futures, preds, decomps, horizon, sample_ids):
    n = len(sample_ids)
    t = np.arange(horizon)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]
    for row, idx in enumerate(sample_ids):
        ax = axes[row]
        gt = futures[idx]
        ax.plot(t, gt, label="GT", color="green", linestyle="--", linewidth=2)
        ax.plot(t, preds[idx], label="Pred", color="black", linewidth=1.5)
        for k, c in zip(["trend", "seasonal", "detail"],
                        ["#e74c3c", "#3498db", "#2ecc71"]):
            ax.plot(t, decomps[k][idx].numpy(), label=k.capitalize(),
                    color=c, linewidth=1.2)
        ax.set_title(f"N-HiTS — Sample {idx}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"N-HiTS Decomposition (h={horizon}, freq_ds=[24,8,1])", fontsize=14
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"nhits_decomp_h{horizon}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = load_ett("ETTh1")
    total_len = len(df)
    print(f"ETTh1: {total_len} rows")

    for h in HORIZONS:
        print(f"\n{'='*60}\n  Horizon = {h}\n{'='*60}")

        # N-HiTS 학습 + 캐시
        mse, mae, preds, decomps = train_and_eval_nhits(
            df=df, target_col="OT",
            train_border=TRAIN_BORDER, val_border=VAL_BORDER,
            context_len=CONTEXT_LEN, horizon=h,
            max_steps=6000,
            cache_path=os.path.join(CACHE_DIR, f"nhits_orig_cache_h{h}.pt"),
        )
        print(f"  MSE={mse:.4f}  MAE={mae:.4f}")

        # 기존 plot_decomposition과 같은 샘플 인덱스 계산
        test_len = total_len - VAL_BORDER
        n_test_windows = test_len - CONTEXT_LEN - h + 1
        # GT 생성 (test 구간에서 future 추출)
        target = df["OT"].values.astype(np.float32)
        futures = np.stack([
            target[VAL_BORDER + i + CONTEXT_LEN : VAL_BORDER + i + CONTEXT_LEN + h]
            for i in range(n_test_windows)
        ])

        sample_ids = np.random.RandomState(42).choice(
            n_test_windows, min(4, n_test_windows), replace=False
        )
        print(f"  Sample IDs: {sample_ids}")

        plot_nhits_decomposition(futures, preds.numpy(), decomps, h, sample_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()
