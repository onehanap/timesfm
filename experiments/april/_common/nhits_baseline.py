"""원본 N-HiTS 베이스라인 (neuralforecast 공식 구현).

이전 구현 `models.nhits_decoder.TimesFMWithNHiTSDecoder` 는 TimesFM 인코더 임베딩
위에 N-HiTS 스타일 디코더를 얹은 **하이브리드**였다. 이는 "같은 backbone에서
디코더만 바꾸는" ablation으로는 의미 있지만, **진정한 N-HiTS 베이스라인은 아니다**
— pretraining의 이득을 공유하기 때문.

이 모듈은 neuralforecast의 NHITS를 사용해 **원본 N-HiTS (원시 시계열 입력,
처음부터 학습)** 를 그대로 베이스라인으로 쓴다.

API:
    mse, mae, preds, decomps = train_and_eval_nhits(
        df, target_col, train_border, val_border,
        context_len, horizon, max_steps=..., cache_path=...,
    )

반환값:
    preds:   torch.Tensor [N_test_windows, horizon]  (ETTDataset 인덱스 순서)
    decomps: dict with keys "trend", "seasonal", "detail"
             각 value shape [N_test_windows, horizon]
             trend + seasonal + detail == preds (within float precision)

Decomposition 추출:
  NHITS는 내부 `decompose_forecast` 플래그 + `BaseModel.decompose(dataset)` 공식
  API를 제공한다. 출력은 [N, n_blocks+1, h] 형태:
    [:, 0, :] = level (Naive1 = 마지막 raw context 값, h 전체에 constant)
    [:, 1, :] = trend stack   (pool=8, freq_downsample=24)
    [:, 2, :] = seasonal stack(pool=4, freq_downsample=8)
    [:, 3, :] = detail stack  (pool=1, freq_downsample=1)

  주의: BaseModel의 _inv_normalization이 [B, 4, h] 입력에 대해 각 stack마다
  독립적으로 `z * scale + loc`을 적용한다. 따라서 stack_i의 "원시 기여분"은
  blocks[:, i, :] - loc 이 된다 (i≥1). loc은 context window의 mean (standard
  scaler 기준). level은 이미 `last_raw_value` 꼴이므로 그대로 쓰면 된다.

  April과 비교 가능한 trend/seasonal/residual 형태로 재구성:
    trend    = level + (blocks[:,1,:] - loc)
    seasonal = blocks[:,2,:] - loc
    detail   = blocks[:,3,:] - loc
  합: trend + seasonal + detail == cv_preds (검증 완료, max_diff=0.0000)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch


def train_and_eval_nhits(
    df: pd.DataFrame,
    target_col: str,
    train_border: int,
    val_border: int,
    context_len: int,
    horizon: int,
    max_steps: int = 1000,
    val_steps_check: int = 100,
    random_seed: int = 0,
    cache_path: str | None = None,
):
    """원본 N-HiTS (neuralforecast)를 학습하고 test 윈도우에서 평가.

    Args:
        df: 전체 시계열 dataframe (train/val/test 연속)
        target_col: 예측 대상 컬럼명 (ETTh1의 경우 "OT")
        train_border: train 끝 index (exclusive) — 보통 8640
        val_border:   val 끝 index (exclusive) — 보통 11520
        context_len:  모델 입력 길이 (NHITS input_size)
        horizon:      예측 길이
        max_steps:    학습 스텝
        val_steps_check: 얼리스타핑 validation 체크 간격
        random_seed:  시드
        cache_path:   결과 캐시 파일 경로 (.pt). 있으면 로드, 없으면 저장.

    Returns:
        (mse, mae, preds_tensor, decomps_dict)
        preds_tensor: torch.Tensor [N_test_windows, horizon]
        decomps_dict: {"trend", "seasonal", "detail"} → each torch.Tensor [N, H]
    """
    # --- 캐시 확인 ---
    if cache_path and os.path.exists(cache_path):
        print(f"  [N-HiTS baseline] 캐시 로드: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "decomps" in cache:
            return cache["mse"], cache["mae"], cache["preds"], cache["decomps"]
        # 구 캐시 (decomps 없음) → 무시하고 재계산
        print(f"  [N-HiTS baseline] 구 캐시 (decomps 없음) 무시, 재계산")

    # --- Import neuralforecast ---
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    # --- Long format 데이터 구성 ---
    series = df[target_col].values.astype(np.float32)
    n = len(series)

    # 임의의 시작 시각 (실제 값은 무관, 규칙적이기만 하면 됨)
    dates = pd.date_range(start="2016-07-01", periods=n, freq="h")
    long_df = pd.DataFrame({
        "unique_id": "series",
        "ds": dates,
        "y": series,
    })

    # --- Test 윈도우 개수 계산 ---
    # ETTDataset(df[val_border:], context_len, horizon) 의 n_samples 와 동일
    test_len = n - val_border
    n_test_windows = test_len - context_len - horizon + 1
    if n_test_windows <= 0:
        raise ValueError(
            f"n_test_windows <= 0: test_len={test_len}, context_len={context_len}, horizon={horizon}"
        )

    # --- Val size (early stopping 용) ---
    # April 실험의 원래 val 구간 = [train_border, val_border)
    # neuralforecast는 val을 "fit 영역의 마지막 val_size행"으로 잡음.
    # Fit 영역 = [0, val_border + context_len) 이 되므로, val 구간을
    # 가능한 한 원래 [train_border, val_border)에 가깝게 맞추기 위해
    # val_size = (val_border - train_border)로 두고 약간의 shift를 감수.
    val_size = val_border - train_border

    # --- Model ---
    print(f"  [N-HiTS baseline] 학습 시작 (h={horizon}, max_steps={max_steps})")
    model = NHITS(
        h=horizon,
        input_size=context_len,
        max_steps=max_steps,
        val_check_steps=min(val_steps_check, max_steps),
        random_seed=random_seed,
        enable_progress_bar=False,
        # 원본 N-HiTS 권장 하이퍼파라미터
        n_pool_kernel_size=[8, 4, 1],
        n_freq_downsample=[24, 8, 1],
        mlp_units=3 * [[512, 512]],
        dropout_prob_theta=0.0,
        activation="ReLU",
        learning_rate=1e-3,
        batch_size=32,
        windows_batch_size=256,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq="h")

    # --- Cross-validation: 1회 fit + rolling prediction ---
    cv_df = nf.cross_validation(
        df=long_df,
        n_windows=n_test_windows,
        step_size=1,
        val_size=val_size,
        refit=False,
        verbose=False,
    )

    # --- CV 결과를 [N_test_windows, horizon] 배열로 변환 ---
    # cv_df 는 cutoff별로 h개 행을 가짐. cutoff 순서가 test_ds 인덱스 순서와 같음.
    cv_df = cv_df.sort_values(["cutoff", "ds"]).reset_index(drop=True)
    unique_cutoffs = cv_df["cutoff"].unique()
    assert len(unique_cutoffs) == n_test_windows, (
        f"expected {n_test_windows} cutoffs, got {len(unique_cutoffs)}"
    )

    preds = cv_df["NHITS"].values.astype(np.float32).reshape(n_test_windows, horizon)
    gts = cv_df["y"].values.astype(np.float32).reshape(n_test_windows, horizon)

    mse = float(np.mean((preds - gts) ** 2))
    mae = float(np.mean(np.abs(preds - gts)))
    preds_tensor = torch.tensor(preds, dtype=torch.float32)

    print(f"  [N-HiTS baseline] MSE={mse:.4f}  MAE={mae:.4f}  (n={n_test_windows})")

    # --- Stack별 decomposition 추출 ---
    # NHITS.decompose(dataset) → [N, n_blocks+1, h], 각 stack이 loc을 더한 inv 상태
    print(f"  [N-HiTS baseline] 분해 추출 중...")
    blocks = nf.models[0].decompose(dataset=nf.dataset, step_size=1)  # np.ndarray

    # 기대 shape: [n_test_windows, 4, horizon] — level + 3 stacks
    if blocks.shape != (n_test_windows, 4, horizon):
        raise RuntimeError(
            f"Unexpected decompose output shape: {blocks.shape}, "
            f"expected ({n_test_windows}, 4, {horizon})"
        )

    # 각 test window의 context mean (= standard scaler loc)
    locs = np.stack([
        series[val_border + i : val_border + i + context_len].mean()
        for i in range(n_test_windows)
    ]).astype(np.float32)                                  # [N]
    locs_b = locs[:, None]                                  # [N, 1]

    # 재구성: 각 stack에서 loc 제거해 원시 기여분 복원 후 April식으로 묶기
    level      = blocks[:, 0, :]                            # 이미 last_raw (loc 더한 상태 아님)
    stack_1    = blocks[:, 1, :] - locs_b                   # trend stack 기여분
    stack_2    = blocks[:, 2, :] - locs_b                   # seasonal stack 기여분
    stack_3    = blocks[:, 3, :] - locs_b                   # detail stack 기여분

    trend    = (level + stack_1).astype(np.float32)
    seasonal = stack_2.astype(np.float32)
    detail   = stack_3.astype(np.float32)

    # Sanity check: trend + seasonal + detail == preds
    recon = trend + seasonal + detail
    max_diff = float(np.abs(recon - preds).max())
    if max_diff > 1e-2:
        print(f"  [N-HiTS baseline] WARNING: decomposition reconstruction diff {max_diff:.4f}")

    decomps = {
        "trend":    torch.tensor(trend,    dtype=torch.float32),
        "seasonal": torch.tensor(seasonal, dtype=torch.float32),
        "detail":   torch.tensor(detail,   dtype=torch.float32),
    }

    # --- 캐시 저장 ---
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(
            {"mse": mse, "mae": mae, "preds": preds_tensor, "decomps": decomps},
            cache_path,
        )
        print(f"  [N-HiTS baseline] 캐시 저장: {cache_path}")

    return mse, mae, preds_tensor, decomps
