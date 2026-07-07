"""Candidate acquisition via official APIs only.

Providers:
- google_cse: Google Programmable Search / Custom Search JSON API, searchType=image.
- openverse: openly licensed images API (no key required, rate-limited).
- local_dir: index an existing folder (offline mode / testing).

Downloads go through PolicyGate; every decision is logged to the manifest.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

from .config import Config
from .policy import PolicyGate, USER_AGENT


@dataclass
class Candidate:
    image_url: str
    thumbnail_url: str | None
    source_page_url: str | None
    title: str | None
    snippet: str | None
    width: int | None
    height: int | None
    mime: str | None
    license: str | None
    api_source: str
    query_used: str
    retrieved_at: float


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 0.01)
        self._last = 0.0

    def wait(self):
        dt = self.min_interval - (time.monotonic() - self._last)
        if dt > 0:
            time.sleep(dt)
        self._last = time.monotonic()


def search_google_cse(cfg: Config, query: str, limit: int) -> list[Candidate]:
    key, cx = cfg.resolve_keys()
    if not key or not cx:
        raise RuntimeError(
            f"Set {cfg.google_cse_key_env} and {cfg.google_cse_cx_env} env vars "
            "(Google Programmable Search JSON API)."
        )
    rl = RateLimiter(cfg.rate_limit_rps)
    out: list[Candidate] = []
    start = 1
    with httpx.Client(timeout=cfg.request_timeout_s, headers={"User-Agent": USER_AGENT}) as c:
        while len(out) < limit and start <= 91:  # CSE caps at 100 results
            rl.wait()
            params = {
                "key": key, "cx": cx, "q": query, "searchType": "image",
                "num": min(cfg.page_size, limit - len(out)), "start": start,
            }
            if cfg.rights_filter:
                params["rights"] = cfg.rights_filter
            r = _get_with_retry(c, "https://www.googleapis.com/customsearch/v1", params, cfg)
            data = r.json()
            items = data.get("items", [])
            if not items:
                break
            for it in items:
                im = it.get("image", {})
                out.append(Candidate(
                    image_url=it.get("link"),
                    thumbnail_url=im.get("thumbnailLink"),
                    source_page_url=im.get("contextLink"),
                    title=it.get("title"),
                    snippet=it.get("snippet"),
                    width=im.get("width"), height=im.get("height"),
                    mime=it.get("mime"),
                    license=None,  # CSE exposes rights only via the rights filter
                    api_source="google_cse", query_used=query,
                    retrieved_at=time.time(),
                ))
            start += len(items)
    return out[:limit]


def search_openverse(cfg: Config, query: str, limit: int) -> list[Candidate]:
    rl = RateLimiter(cfg.rate_limit_rps)
    out: list[Candidate] = []
    page = 1
    with httpx.Client(timeout=cfg.request_timeout_s, headers={"User-Agent": USER_AGENT}) as c:
        while len(out) < limit and page <= 10:
            rl.wait()
            r = _get_with_retry(
                c, "https://api.openverse.org/v1/images/",
                {"q": query, "page_size": min(20, limit - len(out)), "page": page}, cfg,
            )
            results = r.json().get("results", [])
            if not results:
                break
            for it in results:
                out.append(Candidate(
                    image_url=it.get("url"),
                    thumbnail_url=it.get("thumbnail"),
                    source_page_url=it.get("foreign_landing_url"),
                    title=it.get("title"), snippet=None,
                    width=it.get("width"), height=it.get("height"),
                    mime=None,
                    license=f"{it.get('license')} {it.get('license_version') or ''}".strip(),
                    api_source="openverse", query_used=query,
                    retrieved_at=time.time(),
                ))
            page += 1
    return out[:limit]


def search_local_dir(cfg: Config, query: str, limit: int) -> list[Candidate]:
    p = Path(query).expanduser()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    out = []
    for f in sorted(p.rglob("*")):
        if f.suffix.lower() in exts:
            out.append(Candidate(
                image_url=f"file://{f}", thumbnail_url=None, source_page_url=None,
                title=f.name, snippet=None, width=None, height=None, mime=None,
                license="local", api_source="local_dir", query_used=str(p),
                retrieved_at=time.time(),
            ))
        if len(out) >= limit:
            break
    return out


PROVIDERS = {
    "google_cse": search_google_cse,
    "openverse": search_openverse,
    "local_dir": search_local_dir,
}


def _get_with_retry(client: httpx.Client, url: str, params: dict, cfg: Config) -> httpx.Response:
    for attempt in range(cfg.max_retries + 1):
        r = client.get(url, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 2 ** attempt * 2))
            time.sleep(min(wait, 60))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def download(cfg: Config, gate: PolicyGate, cand: Candidate, dest_dir: Path) -> tuple[Path | None, str]:
    """Returns (path, reason). path=None means rejected/failed with reason."""
    if cand.image_url.startswith("file://"):
        src = Path(cand.image_url[7:])
        return (src, "ok") if src.exists() else (None, "missing_local_file")
    d = gate.check_download(cand.image_url, cand.license)
    if not d.allowed:
        return None, d.reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    rl = RateLimiter(cfg.rate_limit_rps)
    try:
        with httpx.Client(timeout=cfg.request_timeout_s, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as c:
            rl.wait()
            r = c.get(cand.image_url)
            if r.status_code != 200:
                return None, f"http_{r.status_code}"
            body = r.content
            if len(body) < cfg.min_bytes:
                return None, "too_small_bytes"
            if len(body) > cfg.max_bytes:
                return None, "too_large_bytes"
            h = hashlib.sha256(body).hexdigest()
            path = dest_dir / f"{h[:20]}{_ext_for(r.headers.get('content-type', ''))}"
            path.write_bytes(body)
            return path, "ok"
    except Exception as e:
        return None, f"error:{type(e).__name__}"


def _ext_for(ct: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(ct.split(";")[0].strip(), ".bin")


def candidate_dict(c: Candidate) -> dict:
    return asdict(c)
