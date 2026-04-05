"""Residual Decoder (with covariate support) — 고주파 직접 예측."""

import torch
import torch.nn as nn


def _build_mlp(input_dim, hidden_layers, activation, dropout):
    act = {"ReLU": nn.ReLU, "SiLU": nn.SiLU, "GELU": nn.GELU}[activation]
    layers = []
    in_d = input_dim
    for h in hidden_layers:
        layers += [nn.Linear(in_d, h), act(), nn.Dropout(dropout)]
        in_d = h
    return nn.Sequential(*layers), in_d


class ResidualDecoder(nn.Module):
    """Flatten → MLP → H 직접 출력 + cov bias."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        horizon: int,
        mlp_units: list[int] | None = None,
        activation: str = "ReLU",
        dropout: float = 0.0,
        cov_fut_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon

        flat_dim = num_patches * embed_dim
        if mlp_units is None:
            mlp_units = [512, 512]
        self.mlp, last_dim = _build_mlp(flat_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, horizon)

        # Full resolution future cov
        self.has_cov = cov_fut_dim > 0
        if self.has_cov:
            self.cov_proj = nn.Linear(cov_fut_dim, 1)

    def forward(self, embeddings: torch.Tensor, cov_future: torch.Tensor | None = None):
        x = embeddings.flatten(1)
        h = self.mlp(x)
        pred = self.forecast_head(h)                     # [B, H]

        if self.has_cov and cov_future is not None:
            cov_bias = self.cov_proj(cov_future).squeeze(-1)  # [B, H]
            pred = pred + cov_bias

        return pred
