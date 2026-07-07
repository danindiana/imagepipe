"""SQLite data model for sessions, searches, images, embeddings, labels, feedback.

Append-only provenance is additionally written to JSONL manifests (see manifest.py);
the DB is the queryable working store.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    intent TEXT,                 -- user natural-language goal
    config_json TEXT NOT NULL    -- frozen copy of effective config
);

CREATE TABLE IF NOT EXISTS searches (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    provider TEXT NOT NULL,      -- e.g. google_cse, openverse, local
    query TEXT NOT NULL,
    params_json TEXT,
    started_at REAL NOT NULL,
    result_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    search_id TEXT REFERENCES searches(id),
    role TEXT NOT NULL DEFAULT 'candidate',  -- candidate | reference | screenshot_crop
    image_url TEXT,
    thumbnail_url TEXT,
    source_page_url TEXT,
    source_domain TEXT,
    title TEXT,
    snippet TEXT,
    license TEXT,                 -- raw license/usage-rights string if known
    api_source TEXT,
    query_used TEXT,
    retrieved_at REAL,
    -- download / file facts
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|downloaded|rejected|quarantined|failed
    reject_reason TEXT,
    original_path TEXT,
    preview_path TEXT,
    mime TEXT,
    width INTEGER, height INTEGER,
    file_size INTEGER,
    sha256 TEXT,
    phash TEXT,
    exif_json TEXT,
    color_json TEXT,              -- dominant colors + histogram summary
    quality_score REAL,           -- resolution/sharpness composite
    -- ranking
    sim_image REAL, sim_text REAL, rank_score REAL,
    cluster_id INTEGER,
    dup_group TEXT,               -- shared id per near-duplicate group
    is_dup_keeper INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_sha ON images(sha256);

CREATE TABLE IF NOT EXISTS embeddings (
    image_id TEXT PRIMARY KEY REFERENCES images(id),
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL          -- float32 little-endian
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL REFERENCES images(id),
    kind TEXT NOT NULL,           -- auto | user | classifier
    label TEXT NOT NULL,          -- e.g. keeper, wrong_object, watermark, illustration
    confidence REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    image_id TEXT NOT NULL REFERENCES images(id),
    signal TEXT NOT NULL,  -- more_like_this|less_like_this|reject|duplicate|favorite|uncertain|keep
    created_at REAL NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def now() -> float:
    return time.time()


def create_session(con, intent: str, config: dict) -> str:
    sid = new_id()
    con.execute(
        "INSERT INTO sessions(id, created_at, intent, config_json) VALUES(?,?,?,?)",
        (sid, now(), intent, json.dumps(config)),
    )
    con.commit()
    return sid
