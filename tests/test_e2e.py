"""End-to-end offline test: synthetic images, local_dir provider, full pipeline.

Run: python tests/test_e2e.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from imagepipe.config import Config
from imagepipe.pipeline import Session
from imagepipe.policy import PolicyGate

TMP = Path("/tmp/imagepipe_test")


def make_images():
    src = TMP / "corpus"
    src.mkdir(parents=True, exist_ok=True)
    def rect(name, bg, fg, jitter=0, size=(640, 480)):
        im = Image.new("RGB", size, bg)
        d = ImageDraw.Draw(im)
        d.rectangle([160 + jitter, 120, 480 + jitter, 360], fill=fg)
        im.save(src / name, quality=92)
    # cluster A: dark rectangles (like "black dial watches")
    rect("a1.jpg", (230, 230, 225), (20, 20, 25))
    rect("a2.jpg", (228, 232, 224), (22, 18, 26), jitter=4)      # near-dup of a1
    shutil.copy(src / "a1.jpg", src / "a1_copy.jpg")             # exact dup
    # cluster B: red circles
    for i, j in enumerate((0, 60)):
        im = Image.new("RGB", (640, 480), (240, 240, 240))
        ImageDraw.Draw(im).ellipse([150 + j, 100, 450 + j, 400], fill=(200, 30, 30))
        im.save(src / f"b{i}.jpg", quality=92)
    # junk: tiny + blank
    Image.new("RGB", (50, 50), (0, 0, 0)).save(src / "tiny.jpg")
    Image.new("RGB", (640, 480), (128, 128, 128)).save(src / "blank.jpg")
    # reference image resembling cluster A
    ref = TMP / "ref.jpg"
    rect(ref.name, (235, 235, 230), (18, 22, 24), jitter=2)
    shutil.move(src / ref.name, ref)
    return src, ref


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    src, ref = make_images()

    cfg = Config(workdir=str(TMP / "work"), provider="local_dir",
                 min_width=200, min_height=200, min_bytes=100)
    s = Session(cfg, intent="dark rectangular object on light background")
    print("session:", s.id)

    # policy gate unit checks
    gate = PolicyGate(cfg)
    assert not gate.check_download("https://encrypted-tbn0.gstatic.com/x.jpg", None).allowed
    cfg.block_domains = ["evil.example"]
    assert not PolicyGate(cfg).check_download("https://a.evil.example/i.jpg", None).allowed
    print("policy gate: OK")

    s.add_reference(ref)
    s.search(str(src), limit=50)
    s.download_pending()
    s.embed_all()
    s.rank(text_prompt=s.intent)

    rows = [dict(r) for r in s.con.execute(
        "SELECT id, image_url, status, reject_reason, rank_score, dup_group, is_dup_keeper "
        "FROM images WHERE session_id=? AND role='candidate' ORDER BY rank_score DESC", (s.id,))]
    ok = [r for r in rows if r["status"] == "downloaded"]
    rej = [r for r in rows if r["status"] == "rejected"]
    print(f"candidates: {len(rows)}, downloaded: {len(ok)}, rejected: {len(rej)}")
    for r in rej:
        print("  rejected:", Path(r['image_url']).name, "->", r["reject_reason"])
    assert any("too_small" in (r["reject_reason"] or "") for r in rej), "tiny not rejected"
    assert any("blank" in (r["reject_reason"] or "") for r in rej), "blank not rejected"

    # dedup: a1/a1_copy/a2 should share a group with one keeper
    groups = {}
    for r in ok:
        groups.setdefault(r["dup_group"], []).append(r)
    big = max(groups.values(), key=len)
    assert len(big) >= 2 and sum(x["is_dup_keeper"] for x in big) == 1, "dedup failed"
    print("dedup: OK (group of", len(big), "with 1 keeper)")

    # ranking: top downloaded candidate should be from cluster A (dark rectangle)
    top = ok[0]
    print("top candidate:", Path(top["image_url"]).name, "score", top["rank_score"])
    assert Path(top["image_url"]).name.startswith("a"), "ranking did not prefer reference-similar images"

    # feedback loop: mark a b-image as more_like_this, expect b images to rise
    b_id = next(r["id"] for r in ok if Path(r["image_url"]).name.startswith("b"))
    before = {r["id"]: r["rank_score"] for r in ok}
    s.feedback(b_id, "more_like_this")
    s.rank(text_prompt=s.intent)
    after = {r["id"]: r["rank_score"] for r in s.con.execute(
        "SELECT id, rank_score FROM images WHERE session_id=? AND status='downloaded' AND role='candidate'", (s.id,))}
    assert after[b_id] > before[b_id], "feedback did not boost score"
    print("relevance feedback: OK")

    out = s.export(strategy="score", top=100)
    assert (out / "export_manifest.csv").exists() and (out / "contact_sheet.html").exists()
    print("export: OK ->", out)

    man = s.dir / "manifest.jsonl"
    events = [json.loads(l)["event"] for l in man.read_text().splitlines()]
    for e in ("search", "candidate", "downloaded", "ranked", "feedback", "export"):
        assert e in events, f"missing manifest event {e}"
    print("provenance manifest: OK (", len(events), "events )")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
