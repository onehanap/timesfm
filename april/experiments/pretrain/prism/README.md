# prism baseline

Prism (multi-frequency seasonal) April variant. Preserved as baseline.

- `prism_pretrain_final.pt` — decoder weights only (~33 MB)
- `cfg.json` — hparams extracted from the ckpt

## Load

```python
import torch
from prism.model.decomp_timesfm_prism import DecompTimesFMPrism  # external project

state = torch.load("prism_pretrain_final.pt", map_location="cpu", weights_only=False)
model = DecompTimesFMPrism(state["cfg"])
model.load_state_dict(state["decoder_state_dict"], strict=False)
```

`DecompTimesFMPrism` is *not* in this repo — it lives in a separate Prism
project. Eval uses `_common.eval_ett_all.build_prism_predictor`.

## Cfg summary

```
ctx_len=1024, batch_size=512, mask_aug_max_patches=16, mask_aug_prob=0.5,
steps_per_stage=6250, n_freq_downsample_trend=32,
n_freq_downsample_seasonal=[16,8,4], corpus="all"
```

## Training code

Not available in this repo. The Prism training script is in the separate
Prism project (`project_prism.md` memory note).
