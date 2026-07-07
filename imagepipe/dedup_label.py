"""Deduplication (phash + embeddings), clustering, and labeling."""
from __future__ import annotations

import numpy as np

from .config import Config


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def dup_groups(cfg: Config, items: list[dict]) -> dict[str, str]:
    """items: [{id, phash, sha256, quality}]. Returns image_id -> group_id.
    Exact dupes (same sha) and near dupes (phash hamming <= threshold) share a group.
    """
    groups: dict[str, str] = {}
    reps: list[tuple[str, str]] = []  # (group_id, phash)
    by_sha: dict[str, str] = {}
    for it in items:
        if it["sha256"] in by_sha:
            groups[it["id"]] = by_sha[it["sha256"]]
            continue
        gid = None
        for g, ph in reps:
            if hamming(it["phash"], ph) <= cfg.dup_phash_hamming:
                gid = g
                break
        if gid is None:
            gid = it["id"]
            reps.append((gid, it["phash"]))
        by_sha[it["sha256"]] = gid
        groups[it["id"]] = gid
    return groups


def pick_keepers(items: list[dict], groups: dict[str, str]) -> set[str]:
    """Highest quality per group wins."""
    best: dict[str, tuple[float, str]] = {}
    for it in items:
        g = groups[it["id"]]
        q = it.get("quality") or 0.0
        if g not in best or q > best[g][0]:
            best[g] = (q, it["id"])
    return {iid for _, iid in best.values()}


def cluster_embeddings(vecs: dict[str, np.ndarray], threshold: float = 0.82) -> dict[str, int]:
    """Greedy leader clustering on cosine similarity (cheap, deterministic)."""
    leaders: list[tuple[int, np.ndarray]] = []
    out: dict[str, int] = {}
    nxt = 0
    for iid, v in vecs.items():
        placed = False
        for cid, lv in leaders:
            if float(v @ lv) >= threshold:
                out[iid] = cid
                placed = True
                break
        if not placed:
            leaders.append((nxt, v))
            out[iid] = nxt
            nxt += 1
    return out


DEFAULT_ZERO_SHOT_LABELS = [
    "a photograph", "an illustration or render", "a diagram or chart",
    "a product photo on white background", "an image with a watermark",
    "a stock photo", "a screenshot",
]


def zero_shot_labels(backend, preview_path: str, prompts: list[str] | None = None,
                     top_k: int = 3) -> list[tuple[str, float]]:
    """CLIP zero-shot labeling; returns [] when the fallback backend is active."""
    prompts = prompts or DEFAULT_ZERO_SHOT_LABELS
    tv = [backend.embed_text(p) for p in prompts]
    if any(t is None for t in tv):
        return []
    iv = backend.embed_image(preview_path)
    sims = [(p, float(iv @ t)) for p, t in zip(prompts, tv)]
    sims.sort(key=lambda x: -x[1])
    return sims[:top_k]
