"""Per-recording segment features, mirroring the official loader's file layout.

Layout under data/features/ (after extracting the Box archives):

    gopro_1s/<backbone>/<recording_id>_360p.mp4_1s_1s.npz   key arr_0, (T, D) float
    gopro_2s/video/<recording_id>_360p.mp4.npz               ImageBind, 2 s segments
    gopro_2s/audio/<recording_id>.wav.npz
    gopro_2s/text/<recording_id>_360p.npz
    gopro_2s/depth/<recording_id>.npz                        (not in the public archive)

ImageBind files store a raw float32 buffer under key ``video_embeddings`` that
reshapes to (T, 1024). The official loader slices rows ``floor(start/L)`` to
``ceil(end/L)`` for each step occurrence, where L is 1 s or 2 s.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"

BACKBONE_DIM = {"omnivore": 1024, "slowfast": 400, "x3d": 400, "3dresnet": 400, "imagebind": 1024}
IMAGEBIND_FILES = {
    "video": "{rid}_360p.mp4.npz",
    "audio": "{rid}.wav.npz",
    "text": "{rid}_360p.npz",
    "depth": "{rid}.npz",
}


class FeatureStore:
    def __init__(self, backbone: str, modality=("video",), root: Path = FEATURES, normalize: bool = False):
        """``normalize`` L2-normalizes every row of every modality before concatenation.

        The released ImageBind video embeddings are unit vectors, but the audio rows
        have norm ~20 and the text rows ~80, so unnormalized concatenation lets audio
        and text dominate the shared encoder.
        """
        if backbone not in BACKBONE_DIM:
            raise ValueError(f"unknown backbone {backbone}")
        self.backbone = backbone
        self.modality = list(modality) if backbone == "imagebind" else ["video"]
        self.segment_len = 2 if backbone == "imagebind" else 1
        self.normalize = normalize
        self.root = Path(root)
        self._cache: dict[str, np.ndarray] = {}
        self.n_clamped = 0  # steps whose time range ran past the feature array

    @property
    def dim(self) -> int:
        return BACKBONE_DIM[self.backbone] * len(self.modality)

    def path(self, rid: str, mod: str = "video") -> Path:
        if self.backbone == "imagebind":
            return self.root / "gopro_2s" / mod / IMAGEBIND_FILES[mod].format(rid=rid)
        return self.root / "gopro_1s" / self.backbone / f"{rid}_360p.mp4_1s_1s.npz"

    def has(self, rid: str) -> bool:
        return all(self.path(rid, m).exists() for m in self.modality)

    def recording(self, rid: str) -> np.ndarray:
        """(T, dim) float32 features for a whole recording, modalities concatenated."""
        if rid in self._cache:
            return self._cache[rid]
        parts = []
        for mod in self.modality:
            with np.load(self.path(rid, mod)) as z:
                if self.backbone == "imagebind":
                    arr = np.frombuffer(z["video_embeddings"], dtype=np.float32).reshape(-1, 1024)
                else:
                    arr = z["arr_0"]
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if self.normalize:
                arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            parts.append(arr)
        if len(parts) > 1:
            t = min(len(p) for p in parts)
            parts = [p[:t] for p in parts]
        feats = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        self._cache[rid] = feats
        return feats

    def step(self, rid: str, occurrences) -> np.ndarray:
        """Rows for a step. ``occurrences`` is a list of (start_s, end_s)."""
        rec = self.recording(rid)
        chunks = []
        for start, end in occurrences:
            a = math.floor(start / self.segment_len)
            b = math.ceil(end / self.segment_len)
            a = min(max(a, 0), len(rec) - 1)
            b = max(min(b, len(rec)), a + 1)
            chunks.append(rec[a:b])
        if any(len(c) == 0 for c in chunks):
            self.n_clamped += 1
        return np.concatenate(chunks, axis=0)

    def preload(self, rids, log=print):
        missing = [r for r in rids if not self.has(r)]
        for r in rids:
            if r not in missing:
                self.recording(r)
        if missing and log:
            log(f"[features] {len(missing)} recordings without {self.backbone}/{'+'.join(self.modality)} features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return missing
