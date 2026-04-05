"""DecompTimesFM with Covariates — TimesFM backbone (frozen) + CovEncoder + 3 decoders."""

import torch
import torch.nn as nn

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import update_running_stats, revin

from .covariate_encoder import CovariateEncoder
from .decoder_t import TrendDecoder
from .decoder_s import SeasonalDecoder
from .decoder_r import ResidualDecoder


class DecompTimesFM(nn.Module):
    """TimesFM (frozen) + CovariateEncoder + Trend/Seasonal/Residual 디코더.

    외인변수는 두 경로로 주입:
      1. Context cov → CovEncoder → concat with backbone emb → 디코더 MLP 입력
      2. Future cov → CovEncoder → 각 디코더에 해상도별 additive bias
    """

    def __init__(self, cfg: dict):
        super().__init__()

        # --- TimesFM backbone (frozen) ---
        pretrained = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=False,
        )
        self.backbone = pretrained.model
        self.patch_len = self.backbone.p    # 32
        self.embed_dim = self.backbone.md   # 1280

        for param in self.backbone.parameters():
            param.requires_grad = False

        context_len = cfg["context_len"]
        horizon = cfg["horizon"]
        num_patches = context_len // self.patch_len
        freq_ds = cfg["n_freq_downsample"]

        cov_ctx_dim = cfg["cov_ctx_dim"]
        cov_fut_dim = cfg["cov_fut_dim"]
        fused_dim = self.embed_dim + cov_ctx_dim  # 1280 + 256 = 1536

        # 디코더별 future cov 차원 (기본: 전부 동일, v2: residual만)
        cov_fut_dims = cfg.get("cov_fut_dims", [cov_fut_dim, cov_fut_dim, cov_fut_dim])

        # --- Covariate Encoder ---
        self.cov_encoder = CovariateEncoder(
            n_cov=cfg["n_cov"],
            num_patches=num_patches,
            cov_ctx_dim=cov_ctx_dim,
            cov_fut_dim=cov_fut_dim,
            channels=cfg["cov_encoder_channels"],
            kernel_size=cfg["cov_encoder_kernel"],
        )

        # --- Decoders (fused_dim as embed_dim) ---
        self.decoder_t = TrendDecoder(
            num_patches=num_patches,
            embed_dim=fused_dim,
            horizon=horizon,
            n_freq_downsample=freq_ds[0],
            mlp_units=cfg["mlp_units"][0],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            cov_fut_dim=cov_fut_dims[0],
        )
        self.decoder_s = SeasonalDecoder(
            num_patches=num_patches,
            embed_dim=fused_dim,
            horizon=horizon,
            n_freq_downsample=freq_ds[1],
            mlp_units=cfg["mlp_units"][1],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            cov_fut_dim=cov_fut_dims[1],
        )
        self.decoder_r = ResidualDecoder(
            num_patches=num_patches,
            embed_dim=fused_dim,
            horizon=horizon,
            mlp_units=cfg["mlp_units"][2],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            cov_fut_dim=cov_fut_dims[2],
        )

        self.context_len = context_len
        self.horizon = horizon

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def _encode(self, context: torch.Tensor, masks: torch.Tensor):
        """target context → (backbone emb, mu, sigma)."""
        B = context.shape[0]
        device = context.device
        num_patches = self.context_len // self.patch_len

        patched_inputs = context.reshape(B, -1, self.patch_len)
        patched_masks = masks.reshape(B, -1, self.patch_len)

        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)
        patch_mu, patch_sigma = [], []
        for i in range(num_patches):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        with torch.no_grad():
            (_, output_embeddings, _, _), _ = self.backbone(normed_inputs, patched_masks)

        last_mu = context_mu[:, -1:]
        last_sigma = context_sigma[:, -1:]
        return output_embeddings, last_mu, last_sigma

    def _encode_and_fuse(self, context, masks, cov_context, cov_future):
        """target + covariates → (fused_emb, cov_fut, mu, sigma)."""
        emb, mu, sigma = self._encode(context, masks)
        cov_ctx, cov_fut = self.cov_encoder(cov_context, cov_future)
        fused = torch.cat([emb, cov_ctx], dim=-1)  # [B, P, 1536]
        return fused, cov_fut, mu, sigma

    def _normalize_future(self, future, mu, sigma):
        return (future - mu) / (sigma + 1e-8)

    # ------------------------------------------------------------------
    # Stage forwards (normalized space)
    # ------------------------------------------------------------------

    def forward_stage1(self, context, masks, cov_context, cov_future):
        fused, cov_fut, mu, sigma = self._encode_and_fuse(
            context, masks, cov_context, cov_future,
        )
        wave_t_n = self.decoder_t(fused, cov_future=cov_fut)
        return wave_t_n, mu, sigma

    def forward_stage2(self, context, masks, cov_context, cov_future):
        fused, cov_fut, mu, sigma = self._encode_and_fuse(
            context, masks, cov_context, cov_future,
        )
        with torch.no_grad():
            wave_t_n = self.decoder_t(fused, cov_future=cov_fut)
        wave_s_n = self.decoder_s(fused, cov_future=cov_fut)
        return wave_t_n, wave_s_n, mu, sigma

    def forward_stage3(self, context, masks, cov_context, cov_future):
        fused, cov_fut, mu, sigma = self._encode_and_fuse(
            context, masks, cov_context, cov_future,
        )
        with torch.no_grad():
            wave_t_n = self.decoder_t(fused, cov_future=cov_fut)
            wave_s_n = self.decoder_s(fused, cov_future=cov_fut)
        residual_n = self.decoder_r(fused, cov_future=cov_fut)
        return wave_t_n, wave_s_n, residual_n, mu, sigma

    # ------------------------------------------------------------------
    # Full inference
    # ------------------------------------------------------------------

    def forward(self, context, masks, cov_context, cov_future):
        fused, cov_fut, mu, sigma = self._encode_and_fuse(
            context, masks, cov_context, cov_future,
        )
        wave_t_n = self.decoder_t(fused, cov_future=cov_fut)
        wave_s_n = self.decoder_s(fused, cov_future=cov_fut)
        residual_n = self.decoder_r(fused, cov_future=cov_fut)

        pred_normed = wave_t_n + wave_s_n + residual_n
        pred = pred_normed * sigma + mu

        wave_t = wave_t_n * sigma + mu
        wave_s = wave_s_n * sigma
        residual = residual_n * sigma

        return pred.squeeze(-1), {
            "trend": wave_t.squeeze(-1),
            "seasonal": wave_s.squeeze(-1),
            "residual": residual.squeeze(-1),
        }

    def get_stage_params(self, stage: int):
        """CovariateEncoder는 모든 stage에서 학습."""
        cov_params = list(self.cov_encoder.parameters())
        if stage == 1:
            return cov_params + list(self.decoder_t.parameters())
        elif stage == 2:
            return cov_params + list(self.decoder_s.parameters())
        elif stage == 3:
            return cov_params + list(self.decoder_r.parameters())
        raise ValueError(f"Invalid stage: {stage}")
