"""TimesFM + 오리지널 N-HiTS 방식 디코더 (외인변수 제외).

원본 N-HiTS 논문의 핵심을 충실히 반영:
  - Block: pooling → MLP → backcast coeffs + forecast coeffs → 보간
  - Stack: 동일 해상도 블록 N개, 잔차(residual) 전달
  - Decoder: 스택 순차 통과, backcast 차감 + forecast 누적
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesfm.timesfm_2p5.timesfm_2p5_torch import (
    TimesFM_2p5_200M_torch,
)
from timesfm.torch.util import update_running_stats, revin


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class NHiTSBlock(nn.Module):
    """단일 N-HiTS 블록.

    1. MaxPool로 임베딩 시퀀스 압축
    2. 평균 풀링 → MLP
    3. forecast 계수 + backcast 계수 출력
    4. 각각 보간하여 horizon / num_patches 길이로 확장
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        pool_kernel: int,
        n_forecast_coeffs: int,
        n_backcast_coeffs: int,
        horizon: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.horizon = horizon
        self.n_forecast_coeffs = n_forecast_coeffs
        self.n_backcast_coeffs = n_backcast_coeffs

        # --- Pooling ---
        if pool_kernel > 1:
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        else:
            self.pool = None

        # --- Shared MLP trunk ---
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Forecast / Backcast heads ---
        self.forecast_head = nn.Linear(hidden_dim, n_forecast_coeffs)
        self.backcast_head = nn.Linear(hidden_dim, n_backcast_coeffs)

        # --- Backcast → embedding 공간 투영 (잔차 차감용) ---
        self.backcast_proj = nn.Linear(1, embed_dim, bias=False)

    def _interpolate(self, coeffs: torch.Tensor, target_len: int) -> torch.Tensor:
        """[B, n_coeffs] → [B, target_len] 선형 보간."""
        if coeffs.shape[1] == target_len:
            return coeffs
        return F.interpolate(
            coeffs.unsqueeze(1), size=target_len, mode="linear", align_corners=False,
        ).squeeze(1)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: [B, num_patches, embed_dim] — 잔차 임베딩

        Returns:
            forecast:       [B, horizon]
            backcast_emb:   [B, num_patches, embed_dim]  (차감할 양)
        """
        # 1) Pooling
        if self.pool is not None:
            x = embeddings.permute(0, 2, 1)       # [B, embed_dim, num_patches]
            x = self.pool(x)                       # [B, embed_dim, pooled]
            x = x.permute(0, 2, 1)                 # [B, pooled, embed_dim]
        else:
            x = embeddings

        # 2) Global average pool → [B, embed_dim]
        x = x.mean(dim=1)

        # 3) MLP
        h = self.mlp(x)                            # [B, hidden_dim]

        # 4) Forecast / Backcast 계수
        forecast_coeffs = self.forecast_head(h)     # [B, n_forecast_coeffs]
        backcast_coeffs = self.backcast_head(h)     # [B, n_backcast_coeffs]

        # 5) 보간
        forecast = self._interpolate(forecast_coeffs, self.horizon)       # [B, horizon]
        backcast = self._interpolate(backcast_coeffs, self.num_patches)   # [B, num_patches]

        # 6) Backcast를 embedding 공간으로 투영
        backcast_emb = self.backcast_proj(backcast.unsqueeze(-1))  # [B, num_patches, embed_dim]

        return forecast, backcast_emb


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

class NHiTSStack(nn.Module):
    """동일 해상도 블록 여러 개를 순차 통과 — 잔차 전달."""

    def __init__(
        self,
        n_blocks: int,
        num_patches: int,
        embed_dim: int,
        pool_kernel: int,
        n_forecast_coeffs: int,
        n_backcast_coeffs: int,
        horizon: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                num_patches=num_patches,
                embed_dim=embed_dim,
                pool_kernel=pool_kernel,
                n_forecast_coeffs=n_forecast_coeffs,
                n_backcast_coeffs=n_backcast_coeffs,
                horizon=horizon,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, embeddings: torch.Tensor):
        """
        Returns:
            stack_forecast: [B, horizon]         — 이 스택의 forecast 합
            residual:       [B, num_patches, embed_dim] — 잔차 임베딩
        """
        residual = embeddings
        stack_forecast = torch.zeros(
            embeddings.shape[0], self.blocks[0].horizon, device=embeddings.device,
        )

        for block in self.blocks:
            forecast, backcast_emb = block(residual)
            residual = residual - backcast_emb
            stack_forecast = stack_forecast + forecast

        return stack_forecast, residual


# ---------------------------------------------------------------------------
# Decoder (전체 N-HiTS 헤드)
# ---------------------------------------------------------------------------

class NHiTSDecoder(nn.Module):
    """오리지널 N-HiTS 방식 디코더: Trend → Seasonal → Detail 스택 순차 통과.

    각 스택은 자신의 해상도에서 설명 가능한 성분을 forecast로 내보내고,
    설명한 만큼을 backcast로 임베딩에서 차감한 뒤 다음 스택에 전달한다.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        horizon: int,
        n_blocks_per_stack: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.horizon = horizon

        # --- 스택별 계수 개수 (원본 N-HiTS식 공격적 분리) ---
        n_trend = max(2, horizon // 48)
        n_seasonal = max(4, horizon // 8)
        n_detail = horizon

        # Backcast 계수: 각 스택이 설명하는 입력 해상도
        nb_trend = max(2, num_patches // 16)
        nb_seasonal = max(4, num_patches // 4)
        nb_detail = num_patches

        self.trend_stack = NHiTSStack(
            n_blocks=n_blocks_per_stack,
            num_patches=num_patches, embed_dim=embed_dim,
            pool_kernel=16,
            n_forecast_coeffs=n_trend, n_backcast_coeffs=nb_trend,
            horizon=horizon, hidden_dim=hidden_dim, dropout=dropout,
        )
        self.seasonal_stack = NHiTSStack(
            n_blocks=n_blocks_per_stack,
            num_patches=num_patches, embed_dim=embed_dim,
            pool_kernel=4,
            n_forecast_coeffs=n_seasonal, n_backcast_coeffs=nb_seasonal,
            horizon=horizon, hidden_dim=hidden_dim, dropout=dropout,
        )
        self.detail_stack = NHiTSStack(
            n_blocks=n_blocks_per_stack,
            num_patches=num_patches, embed_dim=embed_dim,
            pool_kernel=1,
            n_forecast_coeffs=n_detail, n_backcast_coeffs=nb_detail,
            horizon=horizon, hidden_dim=hidden_dim, dropout=dropout,
        )

    def forward(self, output_embeddings: torch.Tensor, return_decomposition: bool = False):
        """
        Args:
            output_embeddings: [B, num_patches, embed_dim]

        Returns:
            pred: [B, horizon]
            (optional) decomp: dict with trend/seasonal/detail
        """
        trend_forecast, residual = self.trend_stack(output_embeddings)
        seasonal_forecast, residual = self.seasonal_stack(residual)
        detail_forecast, _ = self.detail_stack(residual)

        pred = trend_forecast + seasonal_forecast + detail_forecast

        if return_decomposition:
            return pred, {
                "trend": trend_forecast,
                "seasonal": seasonal_forecast,
                "detail": detail_forecast,
            }
        return pred


# ---------------------------------------------------------------------------
# 전체 모델
# ---------------------------------------------------------------------------

class TimesFMWithNHiTSDecoder(nn.Module):
    """TimesFM backbone + 오리지널 N-HiTS 디코더."""

    def __init__(
        self,
        horizon: int,
        context_len: int,
        n_blocks_per_stack: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()

        # 백본 로드
        pretrained = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=False,
        )
        self.backbone = pretrained.model
        self.patch_len = self.backbone.p       # 32
        self.embed_dim = self.backbone.md      # 1280
        self.num_layers = self.backbone.x      # 20
        self.context_len = context_len
        self.horizon = horizon

        num_patches = context_len // self.patch_len

        # 디코더
        self.decoder = NHiTSDecoder(
            num_patches=num_patches,
            embed_dim=self.embed_dim,
            horizon=horizon,
            n_blocks_per_stack=n_blocks_per_stack,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Freeze 전략
        for param in self.backbone.parameters():
            param.requires_grad = False

        for i in range(self.num_layers - unfreeze_last_n, self.num_layers):
            for param in self.backbone.stacked_xf[i].parameters():
                param.requires_grad = True

    def forward(self, context, masks, return_decomposition=False):
        """
        Args:
            context: [B, context_len]
            masks:   [B, context_len]  (True = 패딩)
        """
        B = context.shape[0]
        device = context.device
        num_patches = self.context_len // self.patch_len

        # --- 패치화 ---
        patched_inputs = context.reshape(B, -1, self.patch_len)
        patched_masks = masks.reshape(B, -1, self.patch_len)

        # --- Running stats 정규화 (원본 backbone 방식) ---
        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)
        patch_mu = []
        patch_sigma = []
        for i in range(num_patches):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        # --- 패치별 정규화 ---
        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # --- 백본 forward ---
        (_, output_embeddings, _, _), _ = self.backbone(normed_inputs, patched_masks)

        # --- N-HiTS 디코더 ---
        decoder_out = self.decoder(output_embeddings, return_decomposition)

        # --- Denormalize ---
        last_mu = context_mu[:, -1:]
        last_sigma = context_sigma[:, -1:]

        if return_decomposition:
            normed_pred, decomp = decoder_out
            pred = normed_pred * last_sigma + last_mu
            decomp_denormed = {
                k: v * last_sigma + (last_mu if k == "trend" else 0)
                for k, v in decomp.items()
            }
            return pred, decomp_denormed
        else:
            pred = decoder_out * last_sigma + last_mu
            return pred

    def get_param_groups(self, lr_backbone=1e-5, lr_head=1e-3):
        """백본과 디코더의 learning rate를 분리한 param groups."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        decoder_params = list(self.decoder.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": decoder_params, "lr": lr_head},
        ]
