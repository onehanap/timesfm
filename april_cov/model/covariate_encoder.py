"""Covariate Encoder — 외인변수를 context/future 별로 인코딩."""

import torch
import torch.nn as nn


class CovariateEncoder(nn.Module):
    """외인변수를 Conv1d stack으로 인코딩.

    Returns:
        cov_ctx: [B, num_patches, cov_ctx_dim] — backbone 패치와 정렬
        cov_fut: [B, H, cov_fut_dim] — future 전체 해상도
    """

    def __init__(
        self,
        n_cov: int,
        num_patches: int,
        cov_ctx_dim: int = 256,
        cov_fut_dim: int = 128,
        channels: list[int] | None = None,
        kernel_size: int = 5,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        # Conv1d stack (공유)
        layers = []
        in_ch = n_cov
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        # Context: adaptive pool → patch 정렬
        self.ctx_pool = nn.AdaptiveAvgPool1d(num_patches)
        self.ctx_proj = nn.Linear(channels[-1], cov_ctx_dim)

        # Future: 1x1 conv로 차원 축소
        self.fut_proj = nn.Linear(channels[-1], cov_fut_dim)

    def forward(self, cov_context: torch.Tensor, cov_future: torch.Tensor):
        """
        Args:
            cov_context: [B, C, n_cov]
            cov_future:  [B, H, n_cov]
        Returns:
            cov_ctx: [B, num_patches, cov_ctx_dim]
            cov_fut: [B, H, cov_fut_dim]
        """
        # Conv1d expects [B, channels, time]
        ctx_feat = self.conv_stack(cov_context.permute(0, 2, 1))  # [B, ch, C]
        fut_feat = self.conv_stack(cov_future.permute(0, 2, 1))   # [B, ch, H]

        # Context → patch 정렬
        ctx_pooled = self.ctx_pool(ctx_feat)                       # [B, ch, num_patches]
        cov_ctx = self.ctx_proj(ctx_pooled.permute(0, 2, 1))      # [B, num_patches, cov_ctx_dim]

        # Future → 차원 축소
        cov_fut = self.fut_proj(fut_feat.permute(0, 2, 1))        # [B, H, cov_fut_dim]

        return cov_ctx, cov_fut
