"""디컴포지션 품질 평가 공용 모듈.

모든 april 계열 실험(train.py)에서 공통으로 사용:
    from _common.eval_decomposition import (
        compute_decomposition_metrics,
        print_decomposition_metrics,
        plot_decomposition_metrics,
    )

측정 항목:
  (1) Energy share       — 각 컴포넌트의 분산 비율 (분해 붕괴 탐지)
  (2) STL reference MSE  — future_true에 STL을 돌려 얻은 컴포넌트와의 MSE
  (3) Residual whiteness — Ljung-Box p-value + ACF abs sum
  (6) Cross-component    — |Pearson corr| (orthogonality)
  (7) Partial prediction — T only / T+S / T+S+R MSE
"""

from __future__ import annotations

import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except ImportError:
    _HAS_STL = False

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_LB = True
except ImportError:
    _HAS_LB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pearson_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[N, H] 두 배열의 per-sample Pearson correlation."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1)) + 1e-12
    return num / den


def _acf_abs_sum(x: np.ndarray, max_lag: int) -> float:
    """1..max_lag lag 절대 ACF 합. 낮을수록 white noise에 가까움."""
    x = x - x.mean()
    denom = (x * x).sum() + 1e-12
    s = 0.0
    for lag in range(1, max_lag + 1):
        s += abs(float((x[:-lag] * x[lag:]).sum() / denom))
    return s


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_decomposition_metrics(
    futures,
    decomps,
    period: int = 24,
    stl_max_samples: int = 200,
    lb_max_samples: int = 500,
    acf_max_samples: int = 500,
) -> dict:
    """전체 테스트셋에 대한 디컴포지션 품질 메트릭 딕셔너리를 반환.

    Args:
        futures: [N, H] ground truth
        decomps: dict with keys "trend", "seasonal", "residual", each [N, H]
        period:  seasonal period (ETTh1 hourly → 24)
        stl_max_samples: STL 기준선 계산 시 사용할 샘플 수 (STL은 느리므로 제한)
        lb_max_samples:  Ljung-Box 계산 시 사용할 샘플 수
        acf_max_samples: ACF abs sum 계산 시 사용할 샘플 수
    """
    futures = _to_numpy(futures)
    trend = _to_numpy(decomps["trend"])
    seasonal = _to_numpy(decomps["seasonal"])
    residual = _to_numpy(decomps["residual"])

    assert trend.shape == seasonal.shape == residual.shape == futures.shape, (
        f"shape mismatch: futures={futures.shape}, trend={trend.shape}, "
        f"seasonal={seasonal.shape}, residual={residual.shape}"
    )

    N, H = futures.shape

    # ------------------------------------------------------------------
    # (1) Energy share — per-sample variance ratio, then mean
    # ------------------------------------------------------------------
    var_t = trend.var(axis=1)
    var_s = seasonal.var(axis=1)
    var_r = residual.var(axis=1)
    total = var_t + var_s + var_r + 1e-12
    energy_share = {
        "trend":    float((var_t / total).mean()),
        "seasonal": float((var_s / total).mean()),
        "residual": float((var_r / total).mean()),
    }

    # ------------------------------------------------------------------
    # (2) STL reference MSE — STL(future) 컴포넌트와 모델 출력 비교
    # ------------------------------------------------------------------
    stl_mse = {"trend": None, "seasonal": None, "residual": None, "n_samples": 0}
    if _HAS_STL and period is not None and H >= 2 * period:
        rng = np.random.RandomState(0)
        idx = rng.choice(N, size=min(stl_max_samples, N), replace=False)
        t_errs, s_errs, r_errs = [], [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in idx:
                try:
                    res = STL(futures[i], period=period, robust=True).fit()
                    t_errs.append(float(np.mean((trend[i] - res.trend) ** 2)))
                    s_errs.append(float(np.mean((seasonal[i] - res.seasonal) ** 2)))
                    r_errs.append(float(np.mean((residual[i] - res.resid) ** 2)))
                except Exception:
                    continue
        if t_errs:
            stl_mse = {
                "trend":    float(np.mean(t_errs)),
                "seasonal": float(np.mean(s_errs)),
                "residual": float(np.mean(r_errs)),
                "n_samples": len(t_errs),
            }

    # ------------------------------------------------------------------
    # (3) Residual whiteness — Ljung-Box p-value + ACF abs sum
    # ------------------------------------------------------------------
    lb_pvalue_mean = None
    if _HAS_LB:
        lb_lags = [l for l in [period, 2 * period] if l is not None and l < H]
        if lb_lags:
            pvals = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(min(N, lb_max_samples)):
                    r = residual[i]
                    if r.std() < 1e-8:
                        continue
                    try:
                        lb = acorr_ljungbox(r, lags=lb_lags, return_df=True)
                        pvals.append(float(lb["lb_pvalue"].iloc[-1]))
                    except Exception:
                        continue
            if pvals:
                lb_pvalue_mean = float(np.mean(pvals))

    acf_max_lag = min(period if period else H // 4, H // 4)
    acf_max_lag = max(1, acf_max_lag)
    acf_sums = [
        _acf_abs_sum(residual[i], acf_max_lag)
        for i in range(min(N, acf_max_samples))
    ]
    residual_acf_abs_sum = float(np.mean(acf_sums)) if acf_sums else None

    # ------------------------------------------------------------------
    # (6) Cross-component |Pearson corr|
    # ------------------------------------------------------------------
    cross_corr = {
        "trend_seasonal":    float(np.abs(_pearson_batch(trend, seasonal)).mean()),
        "trend_residual":    float(np.abs(_pearson_batch(trend, residual)).mean()),
        "seasonal_residual": float(np.abs(_pearson_batch(seasonal, residual)).mean()),
    }

    # ------------------------------------------------------------------
    # (7) Partial prediction MSE
    # ------------------------------------------------------------------
    def _mse(a, b):
        return float(np.mean((a - b) ** 2))

    partial_mse = {
        "trend_only":     _mse(trend, futures),
        "trend_seasonal": _mse(trend + seasonal, futures),
        "full":           _mse(trend + seasonal + residual, futures),
    }

    return {
        "energy_share": energy_share,
        "stl_reference_mse": stl_mse,
        "residual_ljung_box_pvalue": lb_pvalue_mean,
        "residual_acf_abs_sum": residual_acf_abs_sum,
        "residual_acf_max_lag": acf_max_lag,
        "cross_corr": cross_corr,
        "partial_mse": partial_mse,
        "period": period,
        "n_test_samples": int(N),
        "horizon": int(H),
    }


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------

def print_decomposition_metrics(metrics: dict, horizon: int) -> None:
    es = metrics["energy_share"]
    stl = metrics["stl_reference_mse"]
    lb = metrics["residual_ljung_box_pvalue"]
    cc = metrics["cross_corr"]
    pm = metrics["partial_mse"]
    acfs = metrics["residual_acf_abs_sum"]

    print(f"\n  [Decomposition Metrics — h={horizon}]")
    print(f"    Energy share   | T={es['trend']:.3f}  S={es['seasonal']:.3f}  R={es['residual']:.3f}")
    if stl["trend"] is not None:
        print(f"    STL ref MSE    | T={stl['trend']:.4f}  S={stl['seasonal']:.4f}  R={stl['residual']:.4f}  (n={stl['n_samples']})")
    else:
        print(f"    STL ref MSE    | N/A (statsmodels missing or H < 2·period)")
    lb_str = f"{lb:.4f}" if lb is not None else "N/A"
    acf_str = f"{acfs:.4f}" if acfs is not None else "N/A"
    print(f"    Residual white | Ljung-Box p={lb_str}   ACF|·|sum(lag≤{metrics['residual_acf_max_lag']})={acf_str}")
    print(f"    |cross corr|   | T-S={cc['trend_seasonal']:.3f}  T-R={cc['trend_residual']:.3f}  S-R={cc['seasonal_residual']:.3f}")
    print(f"    Partial MSE    | T={pm['trend_only']:.4f}  T+S={pm['trend_seasonal']:.4f}  Full={pm['full']:.4f}")


# ---------------------------------------------------------------------------
# Visualization — per-horizon dashboard
# ---------------------------------------------------------------------------

def plot_decomposition_metrics(
    metrics: dict,
    horizon: int,
    output_path: str,
    title_prefix: str = "",
) -> None:
    """2x3 대시보드 PNG 저장."""
    es = metrics["energy_share"]
    stl = metrics["stl_reference_mse"]
    lb = metrics["residual_ljung_box_pvalue"]
    acfs = metrics["residual_acf_abs_sum"]
    cc = metrics["cross_corr"]
    pm = metrics["partial_mse"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    comp_colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # (1) Energy share — bar
    ax = axes[0, 0]
    vals = [es["trend"], es["seasonal"], es["residual"]]
    ax.bar(["trend", "seasonal", "residual"], vals, color=comp_colors)
    ax.set_ylim(0, max(1.0, max(vals) * 1.15))
    ax.set_title("(1) Energy Share (var ratio)")
    ax.set_ylabel("Var(component) / Var(total)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # (2) STL reference MSE — bar
    ax = axes[0, 1]
    if stl["trend"] is not None:
        stl_vals = [stl["trend"], stl["seasonal"], stl["residual"]]
        ax.bar(["trend", "seasonal", "residual"], stl_vals, color=comp_colors)
        ax.set_title(f"(2) STL Reference MSE  (n={stl['n_samples']})")
        ax.set_ylabel("MSE vs STL(future)")
        for i, v in enumerate(stl_vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "STL not computed\n(statsmodels missing\n or H < 2·period)",
                ha="center", va="center", fontsize=11)
        ax.set_title("(2) STL Reference MSE")
    ax.grid(True, alpha=0.3, axis="y")

    # (3) Residual whiteness — textual panel
    ax = axes[0, 2]
    ax.axis("off")
    ax.set_title("(3) Residual Whiteness")
    lb_str = f"{lb:.4f}" if lb is not None else "N/A"
    acf_str = f"{acfs:.4f}" if acfs is not None else "N/A"
    lb_verdict = ""
    if lb is not None:
        lb_verdict = "  ✓ white-ish" if lb > 0.05 else "  ✗ structure left"
    text = (
        f"Ljung-Box p-value (mean)\n"
        f"  = {lb_str}{lb_verdict}\n"
        f"  (↑ high → white noise-like)\n\n"
        f"ACF |·| sum  (lag ≤ {metrics['residual_acf_max_lag']})\n"
        f"  = {acf_str}\n"
        f"  (↓ low → whiter)"
    )
    ax.text(0.02, 0.5, text, fontsize=11, family="monospace", va="center")

    # (6) Cross-correlation heatmap
    ax = axes[1, 0]
    M = np.array([
        [0.0,                   cc["trend_seasonal"], cc["trend_residual"]],
        [cc["trend_seasonal"],  0.0,                  cc["seasonal_residual"]],
        [cc["trend_residual"],  cc["seasonal_residual"], 0.0],
    ])
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1)
    labels = ["trend", "seasonal", "residual"]
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_title("(6) |Cross-correlation|")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.5 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (7) Partial prediction MSE — bar
    ax = axes[1, 1]
    pm_vals = [pm["trend_only"], pm["trend_seasonal"], pm["full"]]
    ax.bar(["T", "T+S", "T+S+R"], pm_vals,
           color=["#e74c3c", "#9b59b6", "#2ecc71"])
    ax.set_title("(7) Partial Prediction MSE")
    ax.set_ylabel("MSE vs ground truth")
    for i, v in enumerate(pm_vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Summary text panel
    ax = axes[1, 2]
    ax.axis("off")
    summary_lines = [
        f"Summary (h={horizon})",
        "─" * 34,
        f"Energy:   T={es['trend']:.2f}  S={es['seasonal']:.2f}  R={es['residual']:.2f}",
    ]
    if stl["trend"] is not None:
        summary_lines.append(
            f"STL MSE:  T={stl['trend']:.3f}  S={stl['seasonal']:.3f}  R={stl['residual']:.3f}"
        )
    summary_lines += [
        f"LB p:     {lb_str}",
        f"ACF sum:  {acf_str}",
        f"|corr|:   T-S={cc['trend_seasonal']:.2f}  T-R={cc['trend_residual']:.2f}  S-R={cc['seasonal_residual']:.2f}",
        f"MSE:      T={pm['trend_only']:.2f}  T+S={pm['trend_seasonal']:.2f}  Full={pm['full']:.2f}",
    ]
    ax.text(0.0, 0.5, "\n".join(summary_lines),
            fontsize=10, family="monospace", va="center")

    fig.suptitle(f"{title_prefix}Decomposition Metrics (h={horizon})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
