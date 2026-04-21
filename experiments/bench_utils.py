"""벤치마크 결과 파일(benchmark_results.txt) 자동 업데이트."""

import os
import re
from datetime import datetime

BENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.txt")


def update_benchmark(model_name: str, results: list[dict], steps_per_stage: int = 0):
    """benchmark_results.txt에서 해당 모델의 결과를 업데이트.

    Args:
        model_name: 모델 이름 (예: "April-SH", "April STL_L1")
        results: [{"horizon":96, "test_mse":..., "test_mae":..., ...}, ...]
        steps_per_stage: 학습 step 수 (stage당)
    """
    if os.path.exists(BENCH_PATH):
        with open(BENCH_PATH, "r") as f:
            content = f.read()
    else:
        content = ""

    # {horizon: {model_name: (mse, mae, steps_info)}}
    horizons_data = {}

    current_h = None
    for line in content.split("\n"):
        h_match = re.match(r"\s*h=(\d+)", line)
        if h_match:
            current_h = int(h_match.group(1))
            if current_h not in horizons_data:
                horizons_data[current_h] = {}
            continue
        if current_h is not None:
            m = re.match(
                r"\s+(.+?)\s*\|\s*MSE:\s*([\d.]+)\s*\|\s*MAE:\s*([\d.]+)(?:\s*\|\s*steps:\s*(.+))?",
                line,
            )
            if m:
                name = m.group(1).strip()
                mse = float(m.group(2))
                mae = float(m.group(3))
                steps = m.group(4).strip() if m.group(4) else ""
                horizons_data[current_h][name] = (mse, mae, steps)

    # 새 결과 반영
    for r in results:
        h = r["horizon"]
        if h not in horizons_data:
            horizons_data[h] = {}
        horizons_data[h]["TimesFM (AR)"] = (r["bl_mse"], r["bl_mae"], "zero-shot")
        horizons_data[h]["N-HiTS"] = (r["nhits_mse"], r["nhits_mae"],
                                      horizons_data[h].get("N-HiTS", ("", "", ""))[2] or f"{steps_per_stage*3}")
        steps_str = f"{steps_per_stage}x3" if steps_per_stage > 0 else ""
        horizons_data[h][model_name] = (r["test_mse"], r["test_mae"], steps_str)

    # 파일 재작성
    lines = []
    lines.append("=" * 60)
    lines.append("  April Benchmark Results — ETTh1 (chronological 60/20/20)")
    lines.append(f"  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    for h in sorted(horizons_data.keys()):
        lines.append("")
        lines.append(f"  h={h}")
        models = horizons_data[h]
        fixed_order = ["TimesFM (AR)", "N-HiTS"]
        preferred_order = ["April L1", "April-Raw", "April Cov_L1", "April Cov_Raw", "April STL_L1"]
        april_models = [k for k in preferred_order if k in models]
        april_models += sorted([k for k in models if k not in fixed_order and k not in april_models])
        for name in fixed_order + april_models:
            if name in models:
                mse, mae, steps = models[name]
                entry = f"    {name:<20s}| MSE: {mse:>8.4f} | MAE: {mae:.4f}"
                if steps:
                    entry += f" | steps: {steps}"
                lines.append(entry)

    lines.append("")

    with open(BENCH_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"  벤치마크 업데이트: {BENCH_PATH}")
