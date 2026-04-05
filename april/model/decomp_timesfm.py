"""DecompTimesFM — TimesFM backbone (frozen) + 3개 분해 디코더."""

import torch
import torch.nn as nn

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import update_running_stats, revin

from .decoder_t import TrendDecoder
from .decoder_s import SeasonalDecoder
from .decoder_r import ResidualDecoder


class DecompTimesFM(nn.Module):
    """
    TimesFM 인코더 (전 구간 freeze) + Trend / Seasonal / Residual 디코더.

    모든 디코더는 normalized space에서 동작한다.
      future_normed ≈ decoder_t + decoder_s + decoder_r
    역정규화는 최종 추론 시에만 수행.
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

        # --- Decoders ---
        self.decoder_t = TrendDecoder(
            num_patches=num_patches,
            embed_dim=self.embed_dim,
            horizon=horizon,
            n_freq_downsample=freq_ds[0],
            mlp_units=cfg["mlp_units"][0],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )
        self.decoder_s = SeasonalDecoder(
            num_patches=num_patches,
            embed_dim=self.embed_dim,
            horizon=horizon,
            n_freq_downsample=freq_ds[1],
            mlp_units=cfg["mlp_units"][1],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )
        self.decoder_r = ResidualDecoder(
            num_patches=num_patches,
            embed_dim=self.embed_dim,
            horizon=horizon,
            mlp_units=cfg["mlp_units"][2],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )

        self.context_len = context_len
        self.horizon = horizon

    # ------------------------------------------------------------------
    # Backbone forward (공통)
    # ------------------------------------------------------------------

    def _encode(self, context: torch.Tensor, masks: torch.Tensor):
        """context → (output_embeddings, last_mu, last_sigma)."""
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

        last_mu = context_mu[:, -1:]      # [B, 1]
        last_sigma = context_sigma[:, -1:]

        return output_embeddings, last_mu, last_sigma

    def _normalize_future(self, future, mu, sigma):
        """future를 normalized space로 변환."""
        return (future - mu) / (sigma + 1e-8)

    # ------------------------------------------------------------------
    # Stage별 forward — 모두 normalized space 출력
    # ------------------------------------------------------------------

    def forward_stage1(self, context, masks):
        """Stage 1: decoder_t 학습. (wave_t_normed, mu, sigma) 반환."""
        emb, mu, sigma = self._encode(context, masks)
        wave_t_normed = self.decoder_t(emb)          # [B, H]
        return wave_t_normed, mu, sigma

    def forward_stage2(self, context, masks):
        """Stage 2: decoder_s 학습. (wave_t_normed, wave_s_normed, mu, sigma) 반환."""
        emb, mu, sigma = self._encode(context, masks)
        with torch.no_grad():
            wave_t_normed = self.decoder_t(emb)
        wave_s_normed = self.decoder_s(emb)
        return wave_t_normed, wave_s_normed, mu, sigma

    def forward_stage3(self, context, masks):
        """Stage 3: decoder_r 학습."""
        emb, mu, sigma = self._encode(context, masks)
        with torch.no_grad():
            wave_t_normed = self.decoder_t(emb)
            wave_s_normed = self.decoder_s(emb)

        residual_normed = self.decoder_r(emb)
        return wave_t_normed, wave_s_normed, residual_normed, mu, sigma

    # ------------------------------------------------------------------
    # 추론 (전체) — 역정규화 포함
    # ------------------------------------------------------------------

    def forward(self, context, masks):
        """전체 추론: wave_t + wave_s + residual, 역정규화 후 반환."""
        emb, mu, sigma = self._encode(context, masks)

        wave_t_normed = self.decoder_t(emb)
        wave_s_normed = self.decoder_s(emb)
        residual_normed = self.decoder_r(emb)

        pred_normed = wave_t_normed + wave_s_normed + residual_normed
        pred = pred_normed * sigma + mu

        # 분해 성분도 역정규화 (trend에만 mu 부여)
        wave_t = wave_t_normed * sigma + mu
        wave_s = wave_s_normed * sigma
        residual = residual_normed * sigma

        return pred.squeeze(-1), {
            "trend": wave_t.squeeze(-1),
            "seasonal": wave_s.squeeze(-1),
            "residual": residual.squeeze(-1),
        }

    def get_stage_params(self, stage: int):
        if stage == 1:
            return self.decoder_t.parameters()
        elif stage == 2:
            return self.decoder_s.parameters()
        elif stage == 3:
            return self.decoder_r.parameters()
        raise ValueError(f"Invalid stage: {stage}")
