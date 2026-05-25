"""gluonts-compatible Predictor wrapping DecompTimesFMPrism.

Adapts prism's `model(context, masks) -> (pred[B, H], decomp)` to the
`predict(test_data_input) -> List[Forecast]` interface that gift-eval uses
via `gluonts.model.evaluate_model`.

Notes:
  - Prism is deterministic (point forecast). We wrap as `QuantileForecast`
    with all quantile levels equal to the point forecast. Point metrics
    (MSE, MAE, MASE, sMAPE, RMSE) are correct; quantile metrics
    (MSIS, mean_weighted_sum_quantile_loss) become degenerate but still
    computable so the pipeline doesn't break.
  - Variable-length contexts: each entry's series is left-padded (or
    truncated to last `context_len`) to match prism's `context_len=1024`.
  - `prediction_length > model.horizon (=128)` is handled by AR rolling
    (append the chunk to context, slide, predict next chunk).
  - flip_invariance: same trick as april — forecast on -ctx and average
    `0.5 * (forecast(ctx) - forecast(-ctx))` to cancel head bias.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm

from timesfm.torch.util import DecodeCache, revin, update_running_stats


_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class PrismPredictor:
    """Gluonts predictor for DecompTimesFMPrism."""

    def __init__(
        self,
        model: torch.nn.Module,
        prediction_length: int,
        context_len: int = 1024,
        device: str = "cuda",
        flip_invariance: bool = True,
    ):
        self.model = model
        self.prediction_length = prediction_length
        self.context_len = context_len
        self.device = device
        self.flip_invariance = flip_invariance
        self.model_h = int(model.horizon)
        self.n_chunks = (prediction_length + self.model_h - 1) // self.model_h

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _build_context_batch(batch_entries, context_len: int):
        """For each entry, take its full series up to now; left-pad with 0
        (mask=True) or truncate to last `context_len` if longer."""
        B = len(batch_entries)
        ctx = np.zeros((B, context_len), dtype=np.float32)
        msk = np.zeros((B, context_len), dtype=bool)
        for i, entry in enumerate(batch_entries):
            arr = np.asarray(entry["target"], dtype=np.float32)
            # NaN safety — gift-eval datasets should be clean, but linear-interp anyway
            if not np.isfinite(arr).all():
                idx = np.arange(arr.size)
                good = np.isfinite(arr)
                if good.any():
                    arr = np.interp(idx, idx[good], arr[good]).astype(np.float32)
                else:
                    arr = np.zeros_like(arr)
            L = arr.size
            if L >= context_len:
                ctx[i] = arr[-context_len:]
            else:
                ctx[i, -L:] = arr
                msk[i, : context_len - L] = True
        return ctx, msk

    @torch.no_grad()
    def _forecast_once(self, ctx_np: np.ndarray, mask_np: np.ndarray):
        """Single direction forecast: returns [B, prediction_length] float32 numpy.

        KV-cached AR rollout, mirroring TimesFM 2.5's native ``decode()``:
        prefill encodes input ONCE through the (frozen) backbone and writes
        every input patch's K/V into a pre-allocated cache; subsequent AR
        steps push only m new patches and reuse the cache.
        """
        backbone = self.model.backbone
        p = backbone.p
        o = int(self.model.horizon)
        m = o // p
        if m * p != o:
            raise ValueError(
                f"model.horizon ({o}) must be a multiple of patch_len ({p})."
            )

        n_layers = backbone.x
        n_heads = backbone.h
        head_dim = backbone.hd

        ctx = torch.from_numpy(ctx_np).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device)
        B = ctx.shape[0]
        num_input_patches = ctx.shape[1] // p
        patched = ctx.reshape(B, num_input_patches, p)
        patched_masks = mask.reshape(B, num_input_patches, p)

        # Prefill: per-patch Welford stats (mask-aware)
        n_s = torch.zeros(B, device=self.device)
        mu_s = torch.zeros(B, device=self.device)
        sg_s = torch.zeros(B, device=self.device)
        mus, sgs = [], []
        for i in range(num_input_patches):
            (n_s, mu_s, sg_s), _ = update_running_stats(
                n_s, mu_s, sg_s, patched[:, i], patched_masks[:, i],
            )
            mus.append(mu_s); sgs.append(sg_s)
        last_n, last_mu, last_sg = n_s, mu_s, sg_s
        patch_mu = torch.stack(mus, dim=1)
        patch_sg = torch.stack(sgs, dim=1)

        # Pre-allocate KV cache to mirror native decode(): without this,
        # backbone(...) with caches=None bypasses cache writes and AR steps
        # lose the input context entirely.
        n_steps = self.n_chunks - 1
        decode_cache_size = num_input_patches + max(n_steps, 0) * m
        caches = [
            DecodeCache(
                next_index=torch.zeros(B, dtype=torch.int32, device=self.device),
                num_masked=torch.zeros(B, dtype=torch.int32, device=self.device),
                key=torch.zeros(B, decode_cache_size, n_heads, head_dim,
                                device=self.device),
                value=torch.zeros(B, decode_cache_size, n_heads, head_dim,
                                  device=self.device),
            )
            for _ in range(n_layers)
        ]

        normed = revin(patched, patch_mu, patch_sg, reverse=False)
        normed = torch.where(patched_masks, 0.0, normed)
        (_, emb, _, _), caches = backbone(normed, patched_masks, caches)
        pred_n = self.model.decode_from_emb(emb)[:, -1]  # [B, o]
        pred = pred_n * patch_sg[:, -1:] + patch_mu[:, -1:]
        chunks = [pred]

        # AR loop: only m new patches per step, KV cache reused
        for _ in range(n_steps):
            new_p = pred.reshape(B, m, p)
            new_msk = torch.zeros_like(new_p, dtype=torch.bool)
            n_s, mu_s, sg_s = last_n, last_mu, last_sg
            new_mus, new_sgs = [], []
            for i in range(m):
                (n_s, mu_s, sg_s), _ = update_running_stats(
                    n_s, mu_s, sg_s, new_p[:, i], new_msk[:, i],
                )
                new_mus.append(mu_s); new_sgs.append(sg_s)
            last_n, last_mu, last_sg = n_s, mu_s, sg_s
            new_mu = torch.stack(new_mus, dim=1)
            new_sg = torch.stack(new_sgs, dim=1)

            new_normed = revin(new_p, new_mu, new_sg, reverse=False)
            (_, new_emb, _, _), caches = backbone(new_normed, new_msk, caches)
            pred_n = self.model.decode_from_emb(new_emb)[:, -1]
            pred = pred_n * new_sg[:, -1:] + new_mu[:, -1:]
            chunks.append(pred)

        full = torch.cat(chunks, dim=1)[:, : self.prediction_length]
        return full.cpu().numpy().astype(np.float32)

    def _forecast_batch(self, batch_entries) -> np.ndarray:
        ctx, msk = self._build_context_batch(batch_entries, self.context_len)
        pos = self._forecast_once(ctx, msk)
        if not self.flip_invariance:
            return pos
        neg = self._forecast_once(-ctx, msk)
        return 0.5 * (pos - neg)

    # ---------------------------------------------------------------
    # gluonts Predictor API
    # ---------------------------------------------------------------

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        self.model.eval()
        forecasts: List[Forecast] = []
        for batch in tqdm(list(batcher(test_data_input, batch_size=batch_size))):
            preds = self._forecast_batch(batch)            # [B, prediction_length]
            for p, entry in zip(preds, batch):
                # Replicate point forecast across all quantiles (degenerate quantile,
                # exact point metrics).
                arr = np.broadcast_to(p[None, :], (len(_QUANTILES), p.shape[0])
                                      ).astype(np.float32)
                forecasts.append(QuantileForecast(
                    forecast_arrays=arr,
                    forecast_keys=[str(q) for q in _QUANTILES],
                    start_date=entry["start"] + len(entry["target"]),
                ))
        return forecasts
