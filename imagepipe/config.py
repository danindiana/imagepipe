"""Configuration loading with conservative defaults."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Config:
    workdir: str = "./imagepipe_data"
    # --- acquisition ---
    provider: str = "google_cse"          # google_cse | openverse | local_dir
    google_cse_key_env: str = "GOOGLE_CSE_KEY"
    google_cse_cx_env: str = "GOOGLE_CSE_CX"
    max_results: int = 60                 # per session, across pages
    page_size: int = 10                   # CSE max
    rate_limit_rps: float = 0.5           # conservative: 1 request / 2s
    download_concurrency: int = 2
    request_timeout_s: float = 20.0
    max_retries: int = 3
    respect_robots: bool = True
    allow_domains: list[str] = field(default_factory=list)   # empty = all not-blocked
    block_domains: list[str] = field(default_factory=list)
    require_license: bool = False         # only keep images with known usage rights
    rights_filter: str = ""               # CSE `rights` param, e.g. cc_publicdomain|cc_attribute
    # --- normalization ---
    min_width: int = 200
    min_height: int = 200
    min_bytes: int = 4096
    max_bytes: int = 30 * 1024 * 1024
    allowed_mime: list[str] = field(default_factory=lambda: ["image/jpeg", "image/png", "image/webp"])
    preview_size: int = 512
    # --- similarity / ranking weights ---
    embed_model: str = "auto"             # auto -> open_clip if installed else color-hist fallback
    w_image_sim: float = 0.45
    w_text_sim: float = 0.25
    w_quality: float = 0.10
    w_license: float = 0.05
    w_source_trust: float = 0.05
    w_feedback: float = 0.10
    dup_phash_hamming: int = 6            # <= means near-duplicate
    # --- ui ---
    ui_host: str = "127.0.0.1"
    ui_port: int = 8787

    def resolve_keys(self):
        return os.environ.get(self.google_cse_key_env), os.environ.get(self.google_cse_cx_env)


def load_config(path: str | None) -> Config:
    cfg = Config()
    if path:
        data = json.loads(Path(path).read_text())
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def dump_config(cfg: Config) -> dict:
    return asdict(cfg)
