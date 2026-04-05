"""Residual Decoder — 고주파 잔차를 direct 예측 (다운샘플 없음)."""

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
    """
    Flatten → MLP → horizon 전체 direct 출력 (pooling/보간 없음).
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        horizon: int,
        mlp_units: list[int] | None = None,
        activation: str = "ReLU",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        flat_dim = num_patches * embed_dim

        if mlp_units is None:
            mlp_units = [512, 512]
        self.mlp, last_dim = _build_mlp(flat_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, horizon)

    def forward(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            embeddings: [B, num_patches, embed_dim]
        Returns:
            residual: [B, horizon]
        """
        x = embeddings.flatten(1)           # [B, P * D]
        h = self.mlp(x)                     # [B, last_dim]
        return self.forecast_head(h)        # [B, horizon]
