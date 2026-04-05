"""Trend Decoder — 저주파 성분을 direct 예측 후 linear 보간."""

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
    """Flatten → MLP → n_coeffs 출력 → linear 보간 → wave_t."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        horizon: int,
        n_freq_downsample: int,
        mlp_units: list[int] | None = None,
        activation: str = "ReLU",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.n_coeffs = max(1, horizon // n_freq_downsample)

        flat_dim = num_patches * embed_dim

        if mlp_units is None:
            mlp_units = [512, 512]
        self.mlp, last_dim = _build_mlp(flat_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, self.n_coeffs)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, num_patches, embed_dim]
        Returns:
            wave_t: [B, horizon]
        """
        x = embeddings.flatten(1)              # [B, P * D]
        h = self.mlp(x)
        coeffs = self.forecast_head(h)         # [B, n_coeffs]

        if self.n_coeffs == self.horizon:
            return coeffs
        wave_t = F.interpolate(
            coeffs.unsqueeze(1),
            size=self.horizon,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return wave_t


def make_trend_target(future: torch.Tensor, pool_kernel: int) -> torch.Tensor:
    """학습 타겟 생성: AvgPool1d로 스무딩 후 linear 보간으로 원래 길이 복원."""
    if pool_kernel <= 1:
        return future
    H = future.shape[1]
    pooled = F.avg_pool1d(future.unsqueeze(1), kernel_size=pool_kernel)
    smoothed = F.interpolate(pooled, size=H, mode="linear", align_corners=False)
    return smoothed.squeeze(1)
