"""Run prism on the GIFT-Eval benchmark and write a CSV in the leaderboard format.

Mirrors the structure of `notebooks/timesfm2p5.ipynb` from the gift-eval repo,
but with `PrismPredictor` instead of `TimesFmPredictor`.

Usage:
  CUDA_VISIBLE_DEVICES=3 python prism/experiments/eval/gifteval/run.py \
      --ckpt prism/experiments/pretrain/prism_pretrain_final.pt \
      [--datasets m4_weekly,m4_monthly]   # default: all
      [--terms short,medium,long]         # default: all (short for univariate-short-only)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Default GIFT-Eval data location
os.environ.setdefault("GIFT_EVAL", "/data1/chaewon/data/useful/gift_eval_bench")

from prism.model.decomp_timesfm_prism import DecompTimesFMPrism      # noqa: E402
from prism.experiments.eval.gifteval.predictor import PrismPredictor  # noqa: E402

from gift_eval.data import Dataset                                    # noqa: E402
from gluonts.ev.metrics import (                                      # noqa: E402
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model                              # noqa: E402
from gluonts.time_feature import get_seasonality                      # noqa: E402

# ---------------------------------------------------------------------------
# Dataset universe (copied verbatim from timesfm2p5.ipynb)
# ---------------------------------------------------------------------------

SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()

MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H "
    "ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Datasets that overlap with prism's LOTSA pretrain (no longer strict zero-shot).
LEAKAGE_DATASETS = {"LOOP_SEATTLE", "M_DENSE", "SZ_TAXI"}


# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=str(PROJECT_ROOT / "prism" / "experiments" /
                                "pretrain" / "prism_pretrain_final.pt"))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out_dir", type=str, default=str(HERE / "results"))
    ap.add_argument("--model_name", type=str, default="prism_pretrain")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--datasets", type=str, default="",
                    help="Comma list to restrict; default = all")
    ap.add_argument("--terms", type=str, default="short,medium,long")
    ap.add_argument("--flip_invariance", type=int, default=1)
    return ap.parse_args()


def build_predictor(ckpt_path: str, device: str, prediction_length: int,
                    flip_invariance: bool) -> PrismPredictor:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = DecompTimesFMPrism(cfg).to(device)
    model.load_state_dict(state["decoder_state_dict"], strict=False)
    model.eval()
    return PrismPredictor(
        model=model,
        prediction_length=prediction_length,
        context_len=cfg.get("context_len", 1024),
        device=device,
        flip_invariance=flip_invariance,
    )


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"Prism ckpt not found: {args.ckpt}\n"
            f"Run pretrain first: python prism/experiments/pretrain/train.py"
        )

    out_dir = Path(args.out_dir) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_results.csv"

    dataset_properties = json.load(
        open(HERE / "dataset_properties.json"),
    )

    if args.datasets:
        wanted = set(args.datasets.split(","))
        all_datasets = [d for d in set(SHORT_DATASETS + MED_LONG_DATASETS) if d in wanted]
    else:
        all_datasets = list(set(SHORT_DATASETS + MED_LONG_DATASETS))
    all_datasets.sort()

    requested_terms = args.terms.split(",")

    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ),
    ]

    header = [
        "dataset", "model",
        "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain", "num_variates", "leakage",
    ]
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for ds_num, ds_name in enumerate(all_datasets):
        print(f"\n[{ds_num+1}/{len(all_datasets)}] {ds_name}")
        for term in ["short", "medium", "long"]:
            if term not in requested_terms:
                continue
            if term in ("medium", "long") and ds_name not in MED_LONG_DATASETS:
                continue

            if "/" in ds_name:
                ds_key, ds_freq = ds_name.split("/")
                ds_key = PRETTY_NAMES.get(ds_key.lower(), ds_key.lower())
            else:
                ds_key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
                ds_freq = dataset_properties[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            try:
                base_dim = Dataset(name=ds_name, term=term, to_univariate=False).target_dim
                dataset = Dataset(name=ds_name, term=term,
                                  to_univariate=base_dim != 1)
            except Exception as e:
                print(f"  [skip] {ds_config}: {e}")
                continue

            season = get_seasonality(dataset.freq)
            print(f"  {ds_config}: pl={dataset.prediction_length} "
                  f"freq={dataset.freq} N={len(dataset.test_data)} "
                  f"season={season}")

            predictor = build_predictor(
                args.ckpt, args.device, dataset.prediction_length,
                flip_invariance=bool(args.flip_invariance),
            )
            res = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=args.batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season,
            )

            leak = ds_name.split("/")[0] in LEAKAGE_DATASETS
            def m(k):
                return res[k].iloc[0]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ds_config, args.model_name,
                    m("MSE[mean]"), m("MSE[0.5]"),
                    m("MAE[0.5]"), m("MASE[0.5]"),
                    m("MAPE[0.5]"), m("sMAPE[0.5]"),
                    m("MSIS"), m("RMSE[mean]"),
                    m("NRMSE[mean]"), m("ND[0.5]"),
                    m("mean_weighted_sum_quantile_loss"),
                    dataset_properties[ds_key]["domain"],
                    dataset_properties[ds_key]["num_variates"],
                    leak,
                ])
            print(f"    → MAE={m('MAE[0.5]'):.4f} MASE={m('MASE[0.5]'):.4f}")

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
