"""Embedding + similarity layer.

Backend is pluggable:
- If open_clip + torch are installed, uses OpenCLIP (ViT-B-32) for real
  image/text embeddings.
- Otherwise falls back to a deterministic color/texture feature vector so the
  full pipeline stays runnable offline (text similarity degrades to 0).

Vector search uses hnswlib when available, else brute-force numpy cosine.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config

# ---------------- backends ----------------

class _ClipBackend:
    name = "openclip_vitb32"
    dim = 512

    def __init__(self):
        import open_clip  # type: ignore
        import torch  # noqa
        self.model, _, self.pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.tok = open_clip.get_tokenizer("ViT-B-32")
        self.model.eval()

    def embed_image(self, path: str) -> np.ndarray:
        import torch
        img = self.pre(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            v = self.model.encode_image(img)[0].numpy()
        return _norm(v.astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            v = self.model.encode_text(self.tok([text]))[0].numpy()
        return _norm(v.astype(np.float32))


class _FallbackBackend:
    """Color-histogram + gradient-orientation features. No text embedding."""
    name = "fallback_colorgrad"
    dim = 128

    def embed_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((96, 96))
        hist = img.histogram()
        color = np.array([sum(hist[i:i + 8]) for i in range(0, 768, 8)], dtype=np.float32)  # 96
        g = np.asarray(img.convert("L"), dtype=np.float32)
        gx = np.diff(g, axis=1)[:-1]
        gy = np.diff(g, axis=0)[:, :-1]
        ang = np.arctan2(gy, gx)
        mag = np.hypot(gx, gy)
        hog, _ = np.histogram(ang, bins=32, range=(-np.pi, np.pi), weights=mag)  # 32
        v = np.concatenate([color / (color.sum() + 1e-6), hog.astype(np.float32) / (hog.sum() + 1e-6)])
        return _norm(v)

    def embed_text(self, text: str) -> np.ndarray | None:
        return None


def get_backend(cfg: Config):
    if cfg.embed_model in ("auto", "openclip"):
        try:
            return _ClipBackend()
        except Exception:
            if cfg.embed_model == "openclip":
                raise
    return _FallbackBackend()


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def vec_to_blob(v: np.ndarray) -> bytes:
    return struct.pack(f"<{len(v)}f", *v.tolist())


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<f4").copy()


# ---------------- index ----------------

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.mat: np.ndarray | None = None
        try:
            import hnswlib  # type: ignore
            self._hnsw = hnswlib.Index(space="cosine", dim=dim)
            self._hnsw.init_index(max_elements=100_000, ef_construction=200, M=16)
        except Exception:
            self._hnsw = None

    def add(self, image_id: str, vec: np.ndarray):
        i = len(self.ids)
        self.ids.append(image_id)
        if self._hnsw is not None:
            self._hnsw.add_items(vec.reshape(1, -1), np.array([i]))
        self.mat = vec.reshape(1, -1) if self.mat is None else np.vstack([self.mat, vec])

    def query(self, vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        k = min(k, len(self.ids))
        if self._hnsw is not None:
            self._hnsw.set_ef(max(64, k))
            labels, dists = self._hnsw.knn_query(vec.reshape(1, -1), k=k)
            return [(self.ids[i], 1.0 - float(d)) for i, d in zip(labels[0], dists[0])]
        sims = self.mat @ vec
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]


# ---------------- ranking ----------------

TRUSTED_TLDS = (".gov", ".edu", ".museum")


def source_trust(domain: str | None) -> float:
    if not domain:
        return 0.5
    if any(domain.endswith(t) for t in TRUSTED_TLDS):
        return 1.0
    return 0.6


def rank_score(cfg: Config, *, sim_image: float, sim_text: float, quality: float,
               has_license: bool, trust: float, feedback: float, dup_penalty: float) -> float:
    s = (cfg.w_image_sim * sim_image + cfg.w_text_sim * sim_text +
         cfg.w_quality * quality + cfg.w_license * (1.0 if has_license else 0.0) +
         cfg.w_source_trust * trust + cfg.w_feedback * feedback)
    return round(max(0.0, s - dup_penalty), 4)


def feedback_vector(pos: list[np.ndarray], neg: list[np.ndarray]) -> np.ndarray | None:
    """Rocchio-style relevance feedback: centroid(pos) - 0.5*centroid(neg)."""
    if not pos:
        return None
    v = np.mean(pos, axis=0)
    if neg:
        v = v - 0.5 * np.mean(neg, axis=0)
    return _norm(v)


def feedback_score(vec: np.ndarray, fb: np.ndarray | None) -> float:
    if fb is None:
        return 0.5
    return float((vec @ fb + 1) / 2)  # map cosine [-1,1] -> [0,1]
