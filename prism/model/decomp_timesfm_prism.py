"""DecompTimesFMPrism — TimesFM 2.5 frozen backbone + 5 ResidualBlock decoders.

Decoders (n_coeffs in parens, horizon=128):
  - trend            : TrendDecoderResi(n_freq_downsample=32)         →  4, linear
  - seasonal_lo      : SeasonalDecoderResi(n_freq_downsample=16)      →  8, cubic
  - seasonal_mid     : SeasonalDecoderResi(n_freq_downsample=8)       → 16, cubic
  - seasonal_hi      : SeasonalDecoderResi(n_freq_downsample=4)       → 32, cubic
  - residual         : ResidualDecoderResi                            → 128, none

Output: pred = wave_t + Σ wave_s_i + wave_r  (normalized space → revin reverse)

April과의 차이:
  - seasonal 디코더가 1개가 아니라 3개 (다주파 분해)
  - get_stage_params가 5단계
"""
from __future__ import annotations

import torch
import torch.nn as nn

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import update_running_stats, revin

from april.model.decoder_resi import (
    TrendDecoderResi, SeasonalDecoderResi, ResidualDecoderResi,
)


class DecompTimesFMPrism(nn.Module):
    SEASONAL_NAMES = ("lo", "mid", "hi")

    def __init__(self, cfg: dict):
        super().__init__()
        pretrained = TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=False,
        )
        self.backbone = pretrained.model
        self.patch_len = self.backbone.p
        self.embed_dim = self.backbone.md
        for p in self.backbone.parameters():
            p.requires_grad = False

        horizon = cfg["horizon"]
        ds_t = cfg["n_freq_downsample_trend"]
        ds_s = cfg["n_freq_downsample_seasonal"]
        if len(ds_s) != 3:
            raise ValueError(
                f"n_freq_downsample_seasonal must have 3 entries, got {len(ds_s)}"
            )
        hidden_dim = cfg.get("hidden_dim", self.embed_dim)
        use_bias = cfg.get("use_bias", False)
        activation = cfg.get("activation", "swish")

        kw = dict(embed_dim=self.embed_dim, horizon=horizon,
                  hidden_dim=hidden_dim, use_bias=use_bias, activation=activation)

        self.decoder_t = TrendDecoderResi(n_freq_downsample=ds_t, **kw)
        self.decoder_s_lo = SeasonalDecoderResi(n_freq_downsample=ds_s[0], **kw)
        self.decoder_s_mid = SeasonalDecoderResi(n_freq_downsample=ds_s[1], **kw)
        self.decoder_s_hi = SeasonalDecoderResi(n_freq_downsample=ds_s[2], **kw)
        self.decoder_r = ResidualDecoderResi(**kw)

        self.context_len = cfg["context_len"]
        self.horizon = horizon

    def _seasonal_decoders(self):
        return (self.decoder_s_lo, self.decoder_s_mid, self.decoder_s_hi)

    # ------------------------------------------------------------------
    # Encoder helpers (per-patch running stats) — april과 동일
    # ------------------------------------------------------------------

    def _encode_to_patches(self, ctx: torch.Tensor, masks: torch.Tensor):
        B = ctx.shape[0]
        device = ctx.device
        P = ctx.shape[1] // self.patch_len

        patched_inputs = ctx.reshape(B, P, self.patch_len)
        patched_masks = masks.reshape(B, P, self.patch_len)

        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)
        mus, sigmas = [], []
        for i in range(P):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i],
            )
            mus.append(mu)
            sigmas.append(sigma)
        patch_mu = torch.stack(mus, dim=1)
        patch_sigma = torch.stack(sigmas, dim=1)

        normed = revin(patched_inputs, patch_mu, patch_sigma, reverse=False)
        normed = torch.where(patched_masks, 0.0, normed)

        with torch.no_grad():
            (_, embeddings, _, _), _ = self.backbone(normed, patched_masks)
        return embeddings, patch_mu, patch_sigma

    # ------------------------------------------------------------------
    # Training: per-patch teacher forcing
    # ------------------------------------------------------------------

    def forward_per_patch(self, ctx: torch.Tensor, masks: torch.Tensor):
        emb, patch_mu, patch_sigma = self._encode_to_patches(ctx, masks)
        wave_t_n = self.decoder_t(emb)
        wave_s_lo_n = self.decoder_s_lo(emb)
        wave_s_mid_n = self.decoder_s_mid(emb)
        wave_s_hi_n = self.decoder_s_hi(emb)
        wave_r_n = self.decoder_r(emb)
        return {
            "emb": emb,
            "wave_t_n": wave_t_n,
            "wave_s_lo_n": wave_s_lo_n,
            "wave_s_mid_n": wave_s_mid_n,
            "wave_s_hi_n": wave_s_hi_n,
            "wave_r_n": wave_r_n,
            "patch_mu": patch_mu, "patch_sigma": patch_sigma,
        }

    # ------------------------------------------------------------------
    # Native-AR helper: sum of normalized decompositions from cached embeddings
    # ------------------------------------------------------------------

    def decode_from_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """Apply all 5 decoders to backbone embeddings (per patch) and sum in
        normalized space. Returns [B, P, horizon]. Used by `forecast_native_ar`
        (KV-cached AR rolling that mirrors TimesFM's `decode()`)."""
        return (self.decoder_t(emb)
                + self.decoder_s_lo(emb)
                + self.decoder_s_mid(emb)
                + self.decoder_s_hi(emb)
                + self.decoder_r(emb))

    # ------------------------------------------------------------------
    # Inference: last-patch forecast
    # ------------------------------------------------------------------

    def forward(self, context: torch.Tensor, masks: torch.Tensor):
        out = self.forward_per_patch(context, masks)
        wt = out["wave_t_n"][:, -1]
        ws_lo = out["wave_s_lo_n"][:, -1]
        ws_mid = out["wave_s_mid_n"][:, -1]
        ws_hi = out["wave_s_hi_n"][:, -1]
        wr = out["wave_r_n"][:, -1]
        mu = out["patch_mu"][:, -1:]
        sigma = out["patch_sigma"][:, -1:]

        pred_n = wt + ws_lo + ws_mid + ws_hi + wr
        pred = pred_n * sigma + mu

        # 시각화용 분해 (역정규화)
        wave_t = wt * sigma + mu
        wave_s_lo = ws_lo * sigma
        wave_s_mid = ws_mid * sigma
        wave_s_hi = ws_hi * sigma
        wave_s_total = wave_s_lo + wave_s_mid + wave_s_hi
        residual = wr * sigma

        return pred, {
            "trend": wave_t,
            "seasonal": wave_s_total,
            "seasonal_lo": wave_s_lo,
            "seasonal_mid": wave_s_mid,
            "seasonal_hi": wave_s_hi,
            "residual": residual,
        }

    # ------------------------------------------------------------------
    # Stage-trainable params (5단계)
    #   1: trend  2: seasonal_lo  3: seasonal_mid  4: seasonal_hi  5: residual
    # ------------------------------------------------------------------

    def get_stage_params(self, stage: int):
        if stage == 1:
            return self.decoder_t.parameters()
        if stage == 2:
            return self.decoder_s_lo.parameters()
        if stage == 3:
            return self.decoder_s_mid.parameters()
        if stage == 4:
            return self.decoder_s_hi.parameters()
        if stage == 5:
            return self.decoder_r.parameters()
        raise ValueError(f"Invalid stage: {stage} (1..5)")

    def get_all_decoder_params(self):
        for d in (self.decoder_t, self.decoder_s_lo, self.decoder_s_mid,
                  self.decoder_s_hi, self.decoder_r):
            yield from d.parameters()
