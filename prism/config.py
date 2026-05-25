"""Prism 모델 설정.

TimesFM 2.5 freeze + 5개 디코더 (trend × 1, seasonal × 3, residual × 1).
대규모 코퍼스 프리트레인 → M3 제로샷 평가용.
"""

DEFAULT_CONFIG = {
    # ── 백본 정렬 ────────────────────────────────────────────
    "context_len": 1024,        # 1024 / 32 = 32 패치
    "horizon": 128,             # TimesFM 2.5 native output_patch_len
    "patch_len": 32,            # TimesFM 2.5 input_patch_len (백본 고정)
    "embed_dim": 1280,          # TimesFM 2.5 model_dims (백본 고정)

    # ── 디코더 분해 비율 ─────────────────────────────────────
    # n_coeffs = horizon // n_freq_downsample
    #   trend          : 4 coeffs (linear interp)
    #   seasonal_lo    : 8 coeffs (cubic interp, 장주기)
    #   seasonal_mid   : 16 coeffs (cubic interp, 중주기)
    #   seasonal_hi    : 32 coeffs (cubic interp, 단주기)
    #   residual       : 128 (no interp, direct horizon)
    "n_freq_downsample_trend": 32,
    "n_freq_downsample_seasonal": [16, 8, 4],

    # ── ResidualBlock 헤드 (TimesFM output_projection_point과 동형) ──
    "hidden_dim": 1280,
    "use_bias": False,
    "activation": "swish",

    # ── 학습 (프리트레인) ────────────────────────────────────
    "learning_rate": 3e-4,
    "warmup_steps": 1000,
    "max_steps_per_stage": 5000,
    "batch_size": 32,
}
