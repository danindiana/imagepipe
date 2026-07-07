"""Session orchestrator tying all stages together, plus JSONL provenance manifests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import acquire, dedup_label, normalize, similarity
from .config import Config, dump_config
from .db import connect, create_session, new_id, now
from .policy import PolicyGate


class Manifest:
    """Append-only JSONL provenance log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, event: str, **kw):
        rec = {"ts": time.time(), "event": event, **kw}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


class Session:
    def __init__(self, cfg: Config, intent: str = "", session_id: str | None = None):
        self.cfg = cfg
        self.root = Path(cfg.workdir)
        self.con = connect(self.root / "imagepipe.db")
        if session_id:
            self.id = session_id
            row = self.con.execute("SELECT intent FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row:
                raise KeyError(f"no such session {session_id}")
            self.intent = row["intent"]
        else:
            self.id = create_session(self.con, intent, dump_config(cfg))
            self.intent = intent
        self.dir = self.root / "sessions" / self.id
        self.manifest = Manifest(self.dir / "manifest.jsonl")
        self.gate = PolicyGate(cfg)
        self._backend = None

    # ---------- backend ----------
    @property
    def backend(self):
        if self._backend is None:
            self._backend = similarity.get_backend(self.cfg)
            self.manifest.log("embed_backend", model=self._backend.name)
        return self._backend

    # ---------- A. inputs ----------
    def add_reference(self, path: Path) -> str:
        return self._register_local(path, role="reference")

    def add_screenshot_crops(self, screenshot: Path, boxes: list[tuple[int, int, int, int]]) -> list[str]:
        """Crop user-indicated regions from a user-provided screenshot.
        Crops are used only as query references, never as bulk-download sources.
        """
        img = Image.open(screenshot).convert("RGB")
        out = []
        crop_dir = self.dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            c = img.crop((x1, y1, x2, y2))
            p = crop_dir / f"crop_{i:03d}.png"
            c.save(p)
            out.append(self._register_local(p, role="screenshot_crop"))
        self.manifest.log("screenshot_crops", screenshot=str(screenshot), boxes=boxes)
        return out

    def _register_local(self, path: Path, role: str) -> str:
        iid = new_id()
        nr = normalize.normalize(self.cfg, path, self.dir / "previews")
        self.con.execute(
            """INSERT INTO images(id, session_id, role, image_url, status, original_path,
               preview_path, mime, width, height, file_size, sha256, phash, exif_json,
               color_json, quality_score, retrieved_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (iid, self.id, role, f"file://{path}", "downloaded" if nr.ok else "rejected",
             str(path), nr.preview_path, nr.mime, nr.width, nr.height, nr.file_size,
             nr.sha256, nr.phash, json.dumps(nr.exif), json.dumps(nr.colors),
             nr.quality, now()))
        self.con.commit()
        self.manifest.log("register_local", image_id=iid, path=str(path), role=role,
                          ok=nr.ok, reason=nr.reason)
        return iid

    # ---------- B. acquisition ----------
    def search(self, query: str, limit: int | None = None) -> str:
        limit = limit or self.cfg.max_results
        fn = acquire.PROVIDERS[self.cfg.provider]
        sid = new_id()
        self.con.execute(
            "INSERT INTO searches(id, session_id, provider, query, params_json, started_at) VALUES(?,?,?,?,?,?)",
            (sid, self.id, self.cfg.provider, query, json.dumps({"limit": limit}), now()))
        cands = fn(self.cfg, query, limit)
        self.manifest.log("search", search_id=sid, provider=self.cfg.provider,
                          query=query, results=len(cands))
        for c in cands:
            iid = new_id()
            dom = None
            if c.image_url and c.image_url.startswith("http"):
                from urllib.parse import urlparse
                dom = urlparse(c.image_url).netloc
            self.con.execute(
                """INSERT INTO images(id, session_id, search_id, role, image_url, thumbnail_url,
                   source_page_url, source_domain, title, snippet, license, api_source,
                   query_used, retrieved_at, width, height, mime)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (iid, self.id, sid, "candidate", c.image_url, c.thumbnail_url,
                 c.source_page_url, dom, c.title, c.snippet, c.license, c.api_source,
                 c.query_used, c.retrieved_at, c.width, c.height, c.mime))
            self.manifest.log("candidate", image_id=iid, **acquire.candidate_dict(c))
        self.con.execute("UPDATE searches SET result_count=? WHERE id=?", (len(cands), sid))
        self.con.commit()
        return sid

    def download_pending(self):
        rows = self.con.execute(
            "SELECT * FROM images WHERE session_id=? AND status='pending' AND role='candidate'",
            (self.id,)).fetchall()
        orig_dir = self.dir / "originals"
        for r in rows:
            c = acquire.Candidate(r["image_url"], r["thumbnail_url"], r["source_page_url"],
                                  r["title"], r["snippet"], r["width"], r["height"],
                                  r["mime"], r["license"], r["api_source"], r["query_used"],
                                  r["retrieved_at"])
            path, reason = acquire.download(self.cfg, self.gate, c, orig_dir)
            if path is None:
                self.con.execute("UPDATE images SET status='rejected', reject_reason=? WHERE id=?",
                                 (reason, r["id"]))
                self.manifest.log("download_rejected", image_id=r["id"], url=r["image_url"], reason=reason)
                continue
            nr = normalize.normalize(self.cfg, path, self.dir / "previews")
            if not nr.ok:
                self.con.execute("UPDATE images SET status='rejected', reject_reason=? WHERE id=?",
                                 (f"normalize:{nr.reason}", r["id"]))
                self.manifest.log("normalize_rejected", image_id=r["id"], reason=nr.reason)
                continue
            self.con.execute(
                """UPDATE images SET status='downloaded', original_path=?, preview_path=?, mime=?,
                   width=?, height=?, file_size=?, sha256=?, phash=?, exif_json=?, color_json=?,
                   quality_score=? WHERE id=?""",
                (str(path), nr.preview_path, nr.mime, nr.width, nr.height, nr.file_size,
                 nr.sha256, nr.phash, json.dumps(nr.exif), json.dumps(nr.colors),
                 nr.quality, r["id"]))
            self.manifest.log("downloaded", image_id=r["id"], path=str(path), sha256=nr.sha256)
        self.con.commit()

    # ---------- D/E/F. embed, rank, dedup ----------
    def embed_all(self):
        rows = self.con.execute(
            """SELECT id, preview_path FROM images WHERE session_id=? AND status='downloaded'
               AND id NOT IN (SELECT image_id FROM embeddings)""", (self.id,)).fetchall()
        for r in rows:
            v = self.backend.embed_image(r["preview_path"])
            self.con.execute("INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?)",
                             (r["id"], self.backend.name, len(v), similarity.vec_to_blob(v)))
        self.con.commit()
        self.manifest.log("embedded", count=len(rows))

    def _vecs(self, role: str | None = None) -> dict[str, np.ndarray]:
        q = """SELECT i.id, e.vector FROM images i JOIN embeddings e ON e.image_id=i.id
               WHERE i.session_id=? AND i.status='downloaded'"""
        args: list = [self.id]
        if role:
            q += " AND i.role=?"
            args.append(role)
        return {r["id"]: similarity.blob_to_vec(r["vector"])
                for r in self.con.execute(q, args)}

    def rank(self, text_prompt: str | None = None):
        refs = {**self._vecs("reference"), **self._vecs("screenshot_crop")}
        cands = self._vecs("candidate")
        if not cands:
            return
        tvec = self.backend.embed_text(text_prompt) if text_prompt else None

        # feedback
        fb_rows = self.con.execute(
            "SELECT image_id, signal FROM feedback WHERE session_id=?", (self.id,)).fetchall()
        all_vecs = {**refs, **cands}
        pos = [all_vecs[r["image_id"]] for r in fb_rows
               if r["signal"] in ("more_like_this", "favorite", "keep") and r["image_id"] in all_vecs]
        neg = [all_vecs[r["image_id"]] for r in fb_rows
               if r["signal"] in ("less_like_this", "reject") and r["image_id"] in all_vecs]
        fb = similarity.feedback_vector(pos, neg)

        ref_mat = np.vstack(list(refs.values())) if refs else None

        # dedup
        items = [dict(r) for r in self.con.execute(
            """SELECT id, phash, sha256, quality_score AS quality FROM images
               WHERE session_id=? AND status='downloaded' AND role='candidate'""", (self.id,))]
        groups = dedup_label.dup_groups(self.cfg, items)
        keepers = dedup_label.pick_keepers(items, groups)
        clusters = dedup_label.cluster_embeddings(cands)

        for iid, v in cands.items():
            sim_img = float(np.max(ref_mat @ v)) if ref_mat is not None else 0.0
            sim_txt = float(tvec @ v) if tvec is not None else 0.0
            row = self.con.execute(
                "SELECT quality_score, license, source_domain FROM images WHERE id=?", (iid,)).fetchone()
            score = similarity.rank_score(
                self.cfg,
                sim_image=(sim_img + 1) / 2, sim_text=(sim_txt + 1) / 2,
                quality=row["quality_score"] or 0.0,
                has_license=bool(row["license"]),
                trust=similarity.source_trust(row["source_domain"]),
                feedback=similarity.feedback_score(v, fb),
                dup_penalty=0.0 if iid in keepers else 0.15)
            self.con.execute(
                """UPDATE images SET sim_image=?, sim_text=?, rank_score=?, dup_group=?,
                   is_dup_keeper=?, cluster_id=? WHERE id=?""",
                (sim_img, sim_txt, score, groups.get(iid), int(iid in keepers),
                 clusters.get(iid), iid))
        self.con.commit()
        self.manifest.log("ranked", candidates=len(cands), refs=len(refs),
                          text_prompt=text_prompt, feedback_pos=len(pos), feedback_neg=len(neg))

    def autolabel(self):
        rows = self.con.execute(
            """SELECT id, preview_path FROM images WHERE session_id=? AND status='downloaded'
               AND role='candidate'""", (self.id,)).fetchall()
        n = 0
        for r in rows:
            labs = dedup_label.zero_shot_labels(self.backend, r["preview_path"])
            for lab, conf in labs:
                self.con.execute(
                    "INSERT INTO labels(image_id, kind, label, confidence, created_at) VALUES(?,?,?,?,?)",
                    (r["id"], "auto", lab, conf, now()))
                n += 1
        self.con.commit()
        self.manifest.log("autolabeled", labels=n)

    # ---------- feedback ----------
    def feedback(self, image_id: str, signal: str):
        self.con.execute(
            "INSERT INTO feedback(session_id, image_id, signal, created_at) VALUES(?,?,?,?)",
            (self.id, image_id, signal, now()))
        self.con.commit()
        self.manifest.log("feedback", image_id=image_id, signal=signal)

    # ---------- H. export ----------
    def export(self, strategy: str = "score", top: int = 200) -> Path:
        out = self.dir / "export"
        rows = [dict(r) for r in self.con.execute(
            """SELECT * FROM images WHERE session_id=? AND status='downloaded'
               AND role='candidate' ORDER BY rank_score DESC LIMIT ?""", (self.id, top))]
        fb = {r["image_id"]: r["signal"] for r in self.con.execute(
            "SELECT image_id, signal FROM feedback WHERE session_id=? ORDER BY created_at",
            (self.id,))}
        import csv
        import shutil
        out.mkdir(parents=True, exist_ok=True)
        man = out / "export_manifest.csv"
        with man.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "folder", "rank_score", "sim_image", "sim_text", "image_url",
                        "source_page_url", "license", "width", "height", "sha256",
                        "dup_group", "cluster_id", "api_source", "query_used"])
            for r in rows:
                folder = _folder_for(strategy, r, fb.get(r["id"]))
                d = out / folder
                d.mkdir(parents=True, exist_ok=True)
                src = Path(r["original_path"])
                if src.exists():
                    shutil.copy2(src, d / src.name)
                w.writerow([r["id"], folder, r["rank_score"], r["sim_image"], r["sim_text"],
                            r["image_url"], r["source_page_url"], r["license"], r["width"],
                            r["height"], r["sha256"], r["dup_group"], r["cluster_id"],
                            r["api_source"], r["query_used"]])
        _contact_sheet(out / "contact_sheet.html", rows, fb)
        self.manifest.log("export", strategy=strategy, count=len(rows), path=str(out))
        return out


def _folder_for(strategy: str, r: dict, fb: str | None) -> str:
    if strategy == "feedback":
        return {"keep": "keeper", "favorite": "keeper", "more_like_this": "keeper",
                "reject": "reject", "less_like_this": "reject", "duplicate": "duplicate",
                "uncertain": "uncertain"}.get(fb or "", "uncertain")
    if strategy == "cluster":
        return f"cluster_{r['cluster_id']}"
    if strategy == "domain":
        return (r["source_domain"] or "unknown").replace(":", "_")
    if strategy == "license":
        return "licensed" if r["license"] else "license_unknown"
    if strategy == "dedup":
        return "keeper" if r["is_dup_keeper"] else "duplicate"
    s = r["rank_score"] or 0
    return "keeper" if s >= 0.6 else ("maybe" if s >= 0.45 else "low_score")


def _contact_sheet(path: Path, rows: list[dict], fb: dict):
    cards = []
    for r in rows:
        prev = Path(r["preview_path"] or "")
        rel = f"../previews/{prev.name}" if prev.name else ""
        cards.append(f"""<div class="card"><img src="{rel}">
<div class="m">score {r['rank_score']} · {r['width']}x{r['height']} · {r['source_domain'] or 'local'}<br>
<a href="{r['source_page_url'] or '#'}" target="_blank">source</a> · fb: {fb.get(r['id'], '-')}
· lic: {r['license'] or '?'}</div></div>""")
    path.write_text(
        "<!doctype html><meta charset=utf-8><style>body{background:#111;color:#ddd;"
        "font:13px sans-serif}div.card{display:inline-block;width:220px;margin:6px;"
        "vertical-align:top}img{width:220px;border-radius:6px}.m{padding:4px 2px}"
        "a{color:#7ab}</style>" + "\n".join(cards))
