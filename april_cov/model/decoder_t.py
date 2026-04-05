"""Trend Decoder (with covariate support) — 저주파 예측 + linear 보간."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim, hidden_layers, activation, dropout):
    act = {"ReLU": nn.ReLU, "SiLU": nn.SiLU, "GELU": nn.GELU}[activation]
    layers = []
    in_d = input_dim
    for h in hidden_layers:
        layers += [nn.Linear(in_d, h), act(), nn.Dropout(dropout)]
        in_d = h
    return nn.Sequential(*layers), in_d


class TrendDecoder(nn.Module):
    """Flatten → MLP → n_coeffs → cov bias → linear 보간."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        horizon: int,
        n_freq_downsample: int,
        mlp_units: list[int] | None = None,
        activation: str = "ReLU",
        dropout: float = 0.0,
        cov_fut_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_coeffs = max(1, horizon // n_freq_downsample)
        self.n_freq_downsample = n_freq_downsample

        flat_dim = num_patches * embed_dim
        if mlp_units is None:
            mlp_units = [512, 512]
        self.mlp, last_dim = _build_mlp(flat_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, self.n_coeffs)

        # Future covariate bias
        self.has_cov = cov_fut_dim > 0
        if self.has_cov:
            self.cov_pool = nn.AdaptiveAvgPool1d(self.n_coeffs)
            self.cov_proj = nn.Linear(cov_fut_dim, 1)

    def forward(self, embeddings: torch.Tensor, cov_future: torch.Tensor | None = None):
        """
        Args:
            embeddings: [B, num_patches, embed_dim] (fused with cov_ctx)
            cov_future: [B, H, cov_fut_dim] or None
        """
        x = embeddings.flatten(1)
        h = self.mlp(x)
        coeffs = self.forecast_head(h)                   # [B, n_coeffs]

        if self.has_cov and cov_future is not None:
            # [B, H, D] → [B, D, H] → pool → [B, D, n_coeffs] → [B, n_coeffs, D]
            cf = self.cov_pool(cov_future.permute(0, 2, 1)).permute(0, 2, 1)
            cov_bias = self.cov_proj(cf).squeeze(-1)     # [B, n_coeffs]
            coeffs = coeffs + cov_bias

        if self.n_coeffs == self.horizon:
            return coeffs
        return F.interpolate(
            coeffs.unsqueeze(1), size=self.horizon, mode="linear", align_corners=False,
        ).squeeze(1)
