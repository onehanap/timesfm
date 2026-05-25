"""Prism pretrain loader — adapted from april's `_common/pretrain_loader.py`
with `DEFAULT_CTX = 1024` (vs april's 512) so daily/hourly subsets contribute
1024-context training windows that match prism's inference setup.

Per-frequency context_len heuristic kept for sampling efficiency only — it
decides which `window_len` to *sample* from each subset (so short series like
yearly Monash data with N≈30 still contribute), but every sampled window is
then left-padded to `MAX_WINDOW_LEN = 1280` so the training script can slice
a uniform `(context_len + horizon) = 1152` window out of every batch element.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import time
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

LOTSA_ROOT = "/data1/chaewon/data/useful/lotsa"
MONASH_ROOT = "/data1/chaewon/data/useful/monash"
MONASH_ZIP_DIR = os.path.join(MONASH_ROOT, "data")
CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "prism_pretrain")

TRAIN_HORIZON = 128
PATCH_LEN = 32
DEFAULT_CTX = 1024                                 # ← april=512, prism=1024
MAX_WINDOW_LEN = DEFAULT_CTX + 2 * TRAIN_HORIZON   # 1280


def freq_to_ctx(freq: str) -> int:
    """monthly+ → 64, weekly → 256, daily- → DEFAULT_CTX."""
    if not freq:
        return DEFAULT_CTX
    f = freq.strip().lower()
    if f in ("yearly", "quarterly", "monthly"):
        return 64
    if f == "weekly":
        return 256
    if f in ("daily", "hourly", "half_hourly", "10_minutes", "minutely", "4_seconds"):
        return DEFAULT_CTX
    fu = freq.strip().upper()
    if fu.startswith(("Y", "A")):
        return 64
    if fu.startswith("Q"):
        return 64
    if fu.startswith("MS") or fu == "M":
        return 64
    if fu.startswith("W"):
        return 256
    return DEFAULT_CTX


def window_len_for_freq(freq: str) -> int:
    return freq_to_ctx(freq) + 2 * TRAIN_HORIZON


# ---------------------------------------------------------------------------
# Subset interface
# ---------------------------------------------------------------------------

class Subset:
    name: str
    n_series: int
    frequency: str
    window_len: int
    def get_length(self, i: int) -> int: ...
    def get_series(self, i: int) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# LOTSA — HuggingFace arrow via datasets.load_from_disk (mmap)
# ---------------------------------------------------------------------------

def _is_lotsa_univariate(info_json_path: str) -> bool:
    with open(info_json_path) as f:
        info = json.load(f)
    tf = info["features"]["target"]
    inner = tf.get("feature", {})
    return "feature" not in inner


def _lotsa_subset_paths(root: str = LOTSA_ROOT) -> List[str]:
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        info_p = os.path.join(p, "dataset_info.json")
        if os.path.isdir(p) and os.path.exists(info_p):
            out.append(p)
    return out


class LotsaSubset(Subset):
    def __init__(self, path: str, cached_lengths: Optional[np.ndarray] = None):
        import datasets as hf_datasets
        self.name = f"lotsa/{os.path.basename(path)}"
        self.path = path
        self._hf = hf_datasets.load_from_disk(path)
        self.n_series = len(self._hf)
        if cached_lengths is not None and len(cached_lengths) == self.n_series:
            self._lengths = cached_lengths
        else:
            tgt_col = self._hf.data["target"]
            lens = np.fromiter(
                (len(v) for v in tgt_col), dtype=np.int64, count=self.n_series,
            )
            self._lengths = lens
        try:
            freq = str(self._hf[0].get("freq", "")) if self.n_series > 0 else ""
        except Exception:
            freq = ""
        self.frequency = freq
        self.window_len = window_len_for_freq(freq)

    def get_length(self, i: int) -> int:
        return int(self._lengths[i])

    def get_series(self, i: int) -> np.ndarray:
        row = self._hf[int(i)]
        return np.asarray(row["target"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Monash — parsed from .tsf into in-memory list[np.ndarray]
# ---------------------------------------------------------------------------

def _parse_tsf_bytes(content: bytes) -> Tuple[List[np.ndarray], str]:
    series_list: List[np.ndarray] = []
    frequency = ""
    in_data = False
    text = content.decode("cp1252", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            if line.startswith("@frequency"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    frequency = parts[1].strip()
            elif line.startswith("@data"):
                in_data = True
            continue
        if not in_data:
            continue
        parts = line.split(":")
        raw = parts[-1].split(",")
        vals: List[float] = []
        for v in raw:
            if v == "?":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size < 2:
            continue
        if np.isnan(arr).any():
            mask = np.isnan(arr)
            if mask.all():
                continue
            idx = np.arange(arr.size)
            arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
        series_list.append(arr)
    return series_list, frequency


def _monash_zip_paths(root: str = MONASH_ZIP_DIR) -> List[str]:
    return sorted(
        os.path.join(root, n) for n in os.listdir(root) if n.endswith(".zip")
    )


def _read_tsf_frequency(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path) as z:
        tsf_name = next(n for n in z.namelist() if n.endswith(".tsf"))
        with z.open(tsf_name) as fp:
            for raw in fp:
                try:
                    line = raw.decode("cp1252", errors="replace").strip()
                except Exception:
                    continue
                if line.startswith("@frequency"):
                    parts = line.split(None, 1)
                    return parts[1].strip() if len(parts) == 2 else ""
                if line.startswith("@data"):
                    break
    return ""


class MonashSubset(Subset):
    def __init__(self, zip_path: str, cache_dir: str = CACHE_ROOT):
        self.zip_path = zip_path
        base = os.path.basename(zip_path).replace(".zip", "")
        self.name = f"monash/{base}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_p = os.path.join(cache_dir, f"monash_{base}.pkl")
        if os.path.exists(cache_p):
            with open(cache_p, "rb") as f:
                self.series = pickle.load(f)
            try:
                freq = _read_tsf_frequency(zip_path)
            except Exception:
                freq = ""
        else:
            with zipfile.ZipFile(zip_path) as z:
                tsf_name = next(n for n in z.namelist() if n.endswith(".tsf"))
                with z.open(tsf_name) as fp:
                    data = fp.read()
            self.series, freq = _parse_tsf_bytes(data)
            with open(cache_p, "wb") as f:
                pickle.dump(self.series, f)
        self.frequency = freq
        self.window_len = window_len_for_freq(freq)
        self.n_series = len(self.series)
        self._lengths = np.fromiter(
            (s.size for s in self.series), dtype=np.int64, count=self.n_series,
        )

    def get_length(self, i: int) -> int:
        return int(self._lengths[i])

    def get_series(self, i: int) -> np.ndarray:
        return self.series[int(i)]


# ---------------------------------------------------------------------------
# Enumerate and filter
# ---------------------------------------------------------------------------

MIN_SERIES_PER_SUBSET = 10
# Series shorter than this are dropped at the series level. Series between
# MIN_SERIES_LEN and window_len are KEPT and left-padded at sample time
# (matches TimesFM pretrain handling instead of throwing the data away).
MIN_SERIES_LEN = TRAIN_HORIZON + PATCH_LEN  # 128 + 32 = 160


def enumerate_lotsa(root: str = LOTSA_ROOT) -> List[LotsaSubset]:
    subsets: List[LotsaSubset] = []
    for p in _lotsa_subset_paths(root):
        info_p = os.path.join(p, "dataset_info.json")
        if not _is_lotsa_univariate(info_p):
            continue
        try:
            s = LotsaSubset(p)
        except Exception as e:
            print(f"  [warn] failed to open LOTSA subset {p}: {e}")
            continue
        n_usable = int(np.sum(s._lengths >= MIN_SERIES_LEN))
        if n_usable < MIN_SERIES_PER_SUBSET:
            continue
        subsets.append(s)
    return subsets


def enumerate_monash(root: str = MONASH_ZIP_DIR) -> List[MonashSubset]:
    subsets: List[MonashSubset] = []
    for p in _monash_zip_paths(root):
        try:
            s = MonashSubset(p)
        except Exception as e:
            print(f"  [warn] failed to open Monash {p}: {e}")
            continue
        n_usable = int(np.sum(s._lengths >= MIN_SERIES_LEN))
        if n_usable < MIN_SERIES_PER_SUBSET:
            continue
        subsets.append(s)
    return subsets


def build_subsets() -> List[Subset]:
    t0 = time.time()
    lotsa = enumerate_lotsa()
    print(f"  LOTSA usable subsets: {len(lotsa)} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    monash = enumerate_monash()
    print(f"  Monash usable subsets: {len(monash)} ({time.time()-t0:.1f}s)")
    return lotsa + monash


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class SubsetSampler:
    """Series with length >= MIN_SERIES_LEN are eligible. Short series (< wl)
    are still sampled — caller takes the entire series and the loader left-pads
    to MAX_WL at yield time."""

    def __init__(self, subsets: Sequence[Subset]):
        self.subsets = list(subsets)
        self.usable: List[np.ndarray] = []
        self.lengths: List[np.ndarray] = []
        self.win_lens: List[int] = []
        for s in self.subsets:
            wl = int(s.window_len)
            idx = np.nonzero(s._lengths >= MIN_SERIES_LEN)[0].astype(np.int64)
            self.usable.append(idx)
            self.lengths.append(s._lengths.copy())
            self.win_lens.append(wl)

    def sample(self, rng: random.Random) -> Tuple[int, int, int, int, int]:
        """Returns (si, seri, start, wl, actual_len)."""
        si = rng.randrange(len(self.subsets))
        usable = self.usable[si]
        seri = int(usable[rng.randrange(usable.size)])
        wl = self.win_lens[si]
        L = int(self.lengths[si][seri])
        if L >= wl:
            max_start = L - wl
            start = rng.randint(0, max_start)
            actual_len = wl
        else:
            start = 0
            actual_len = L
        return si, seri, start, wl, actual_len


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------

class PretrainDataset(IterableDataset):
    MAX_ABS = 1e6
    MIN_PATCH_STD = 1e-3
    PATCH_LEN = PATCH_LEN
    MAX_WL = MAX_WINDOW_LEN

    def __init__(self, subsets: Sequence[Subset], seed: int = 0):
        super().__init__()
        self.sampler = SubsetSampler(subsets)
        self.seed = seed
        assert self.MAX_WL % self.PATCH_LEN == 0
        for s in subsets:
            assert s.window_len % self.PATCH_LEN == 0, \
                f"{s.name}: window_len {s.window_len} not patch-aligned"
            assert s.window_len <= self.MAX_WL, \
                f"{s.name}: window_len {s.window_len} > MAX_WL {self.MAX_WL}"

    def _is_good_window(self, window: np.ndarray) -> bool:
        """Quality check on the *real* slice. Trims to the largest whole-patch
        suffix so std check stays valid for short (< wl) series."""
        if window.size < self.PATCH_LEN:
            return False
        if not np.isfinite(window).all():
            return False
        if np.max(np.abs(window)) > self.MAX_ABS:
            return False
        n_patches = window.size // self.PATCH_LEN
        trimmed = window[-(n_patches * self.PATCH_LEN):]
        patched = trimmed.reshape(n_patches, self.PATCH_LEN)
        stds = patched.std(axis=1)
        if (stds < self.MIN_PATCH_STD).any():
            return False
        return True

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        winfo = get_worker_info()
        if winfo is None:
            rng = random.Random(self.seed)
        else:
            rng = random.Random(self.seed + winfo.id * 100003)
        sampler = self.sampler
        max_tries = 16
        while True:
            tries = 0
            actual_len = 0
            window = None
            while tries < max_tries:
                si, seri, start, wl, actual_len = sampler.sample(rng)
                series = sampler.subsets[si].get_series(seri)
                window = np.asarray(series[start: start + actual_len], dtype=np.float32)
                if window.size == actual_len and self._is_good_window(window):
                    break
                tries += 1
            else:
                actual_len = self.MAX_WL
                window = np.linspace(-1.0, 1.0, actual_len, dtype=np.float32) + \
                    np.random.normal(size=actual_len).astype(np.float32) * 0.01

            padded = np.zeros(self.MAX_WL, dtype=np.float32)
            mask = np.zeros(self.MAX_WL, dtype=bool)
            # Left-pad: real data RIGHT-aligned (last `actual_len` positions),
            # mask=True for all preceding pad positions.
            padded[-actual_len:] = window
            if actual_len < self.MAX_WL:
                mask[: self.MAX_WL - actual_len] = True
            yield torch.from_numpy(padded.copy()), torch.from_numpy(mask.copy())


def build_dataloader(
    batch_size: int,
    num_workers: int = 4,
    subsets: Optional[Sequence[Subset]] = None,
    seed: int = 0,
) -> Tuple[DataLoader, List[Subset]]:
    if subsets is None:
        subsets = build_subsets()
    ds = PretrainDataset(subsets, seed=seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return dl, list(subsets)
