"""April 모델 추론 — 학습된 체크포인트로 예측 및 분해 시각화."""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.decomp_timesfm import DecompTimesFM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "april")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = DecompTimesFM(cfg).to(device)

    # 디코더 가중치만 로드 (backbone은 pretrained에서 이미 로드됨)
    decoder_state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(decoder_state, strict=False)
    # backbone 키는 missing이 정상
    missing_non_backbone = [k for k in missing if not k.startswith("backbone.")]
    if missing_non_backbone:
        print(f"Warning: missing decoder keys: {missing_non_backbone}")

    model.eval()
    return model, cfg


@torch.no_grad()
def predict(model, context: np.ndarray, device) -> dict:
    """단일 시계열 예측.

    Args:
        model: 학습된 DecompTimesFM
        context: [context_len] numpy array
        device: torch device

    Returns:
        dict with keys: prediction, trend, seasonal, residual (all numpy)
    """
    ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    masks = torch.zeros(1, len(context), dtype=torch.bool, device=device)

    pred, decomp = model(ctx, masks)

    return {
        "prediction": pred[0].cpu().numpy(),
        "trend": decomp["trend"][0].cpu().numpy(),
        "seasonal": decomp["seasonal"][0].cpu().numpy(),
        "residual": decomp["residual"][0].cpu().numpy(),
    }


@torch.no_grad()
def predict_batch(model, contexts: np.ndarray, device) -> dict:
    """배치 예측.

    Args:
        contexts: [B, context_len] numpy array
    Returns:
        dict with keys: prediction, trend, seasonal, residual (all [B, H] numpy)
    """
    ctx = torch.tensor(contexts, dtype=torch.float32).to(device)
    masks = torch.zeros_like(ctx, dtype=torch.bool)

    pred, decomp = model(ctx, masks)

    return {
        "prediction": pred.cpu().numpy(),
        "trend": decomp["trend"].cpu().numpy(),
        "seasonal": decomp["seasonal"].cpu().numpy(),
        "residual": decomp["residual"].cpu().numpy(),
    }


def plot_single(result, future=None, save_path=None):
    """단일 예측 결과 시각화."""
    horizon = len(result["prediction"])
    t = np.arange(horizon)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 전체 예측
    ax = axes[0]
    if future is not None:
        ax.plot(t, future, label="Ground Truth", color="green", linestyle="--", linewidth=2)
    ax.plot(t, result["prediction"], label="Prediction", color="black", linewidth=1.5)
    ax.set_title("Total Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trend
    ax = axes[1]
    ax.plot(t, result["trend"], color="#e74c3c", linewidth=1.5)
    ax.set_title("Trend (wave_t)")
    ax.grid(True, alpha=0.3)

    # Seasonal
    ax = axes[2]
    ax.plot(t, result["seasonal"], color="#3498db", linewidth=1.5)
    ax.set_title("Seasonal (wave_s)")
    ax.grid(True, alpha=0.3)

    # Residual
    ax = axes[3]
    ax.plot(t, result["residual"], color="#2ecc71", linewidth=1.5)
    ax.set_title("Residual")
    ax.set_xlabel("Time Step")
    ax.grid(True, alpha=0.3)

    plt.suptitle("April Model — Decomposed Forecast", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="April Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="ETTh2",
                        help="ETT dataset name for demo")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Sample index for visualization")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, device)
    horizon = cfg["horizon"]
    context_len = cfg["context_len"]
    print(f"Context: {context_len}, Horizon: {horizon}")

    # Load data
    import pandas as pd
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{args.dataset}.csv"
    df = pd.read_csv(url)
    values = df["OT"].values.astype(np.float32)

    idx = args.sample_idx
    context = values[idx : idx + context_len]
    future = values[idx + context_len : idx + context_len + horizon]

    if len(context) < context_len or len(future) < horizon:
        print(f"Error: Not enough data at index {idx}")
        return

    # Predict
    result = predict(model, context, device)
    mse = np.mean((result["prediction"] - future) ** 2)
    mae = np.mean(np.abs(result["prediction"] - future))
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot
    save_path = os.path.join(OUTPUT_DIR, f"inference_h{horizon}_idx{idx}.png")
    plot_single(result, future=future, save_path=save_path)


if __name__ == "__main__":
    main()
