"""Image normalization: signature check, EXIF orientation, previews, phash, colors."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Config

SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # + 'WEBP' at offset 8, checked below
}


@dataclass
class NormResult:
    ok: bool
    reason: str = ""
    mime: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    sha256: str = ""
    phash: str = ""
    exif: dict | None = None
    colors: dict | None = None
    quality: float = 0.0
    preview_path: str = ""


def sniff_mime(data: bytes) -> str | None:
    for sig, mime in SIGNATURES.items():
        if data.startswith(sig):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return None


def normalize(cfg: Config, original: Path, preview_dir: Path) -> NormResult:
    try:
        data = original.read_bytes()
    except OSError:
        return NormResult(False, "unreadable")
    mime = sniff_mime(data)
    if mime is None or mime not in cfg.allowed_mime:
        return NormResult(False, f"bad_signature:{mime}")
    try:
        img = Image.open(original)
        img.load()
    except (UnidentifiedImageError, OSError):
        return NormResult(False, "corrupt")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if w < cfg.min_width or h < cfg.min_height:
        return NormResult(False, f"too_small:{w}x{h}")
    rgb = img.convert("RGB")
    # blank detection: near-zero variance
    small = rgb.resize((32, 32))
    px = list(small.getdata())
    mean = [sum(c[i] for c in px) / len(px) for i in range(3)]
    var = sum(sum((c[i] - mean[i]) ** 2 for i in range(3)) for c in px) / len(px)
    if var < 15:
        return NormResult(False, "blank")

    preview_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(data).hexdigest()
    prev = preview_dir / f"{sha[:20]}.jpg"
    thumb = rgb.copy()
    thumb.thumbnail((cfg.preview_size, cfg.preview_size))
    thumb.save(prev, "JPEG", quality=85)

    ph = str(imagehash.phash(rgb))
    exif = {}
    try:
        raw = img.getexif()
        exif = {str(k): str(v)[:200] for k, v in list(raw.items())[:40]}
    except Exception:
        pass
    colors = _color_stats(rgb)
    quality = _quality_score(rgb, w, h)
    return NormResult(True, "ok", mime, w, h, len(data), sha, ph, exif, colors, quality, str(prev))


def _color_stats(rgb: Image.Image) -> dict:
    small = rgb.resize((64, 64))
    q = small.quantize(colors=5, method=Image.Quantize.MEDIANCUT).convert("RGB")
    counts: dict[tuple, int] = {}
    for p in q.getdata():
        counts[p] = counts.get(p, 0) + 1
    dom = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    total = 64 * 64
    hist = small.histogram()  # 768 bins
    coarse = [sum(hist[i:i + 32]) for i in range(0, 768, 32)]
    return {
        "dominant": [{"rgb": list(c), "frac": round(n / total, 3)} for c, n in dom],
        "hist24": coarse,
    }


def _quality_score(rgb: Image.Image, w: int, h: int) -> float:
    """Composite of resolution and a cheap sharpness proxy (edge energy)."""
    res = min(1.0, (w * h) / (1600 * 1200))
    g = rgb.convert("L").resize((128, 128))
    px = list(g.getdata())
    edge = 0
    for y in range(127):
        row, nxt = y * 128, (y + 1) * 128
        for x in range(127):
            edge += abs(px[row + x] - px[row + x + 1]) + abs(px[row + x] - px[nxt + x])
    sharp = min(1.0, edge / (127 * 127 * 2) / 25.0)
    return round(0.6 * res + 0.4 * sharp, 4)


def to_json(nr: NormResult) -> str:
    return json.dumps(nr.__dict__)
