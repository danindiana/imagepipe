"""Local human-in-the-loop review UI (FastAPI, single page)."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import Config
from .pipeline import Session

PAGE = """<!doctype html><meta charset=utf-8><title>imagepipe review</title>
<style>
body{background:#101216;color:#d6d9de;font:14px system-ui;margin:0;padding:16px}
h1{font-size:16px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.card{background:#1a1e25;border-radius:10px;padding:8px}
.card img{width:100%;border-radius:6px;aspect-ratio:1;object-fit:contain;background:#0b0d10}
.meta{font-size:12px;color:#9aa1ab;margin:6px 0} .meta a{color:#7fb2d9}
.btns button{margin:2px;padding:4px 8px;border:0;border-radius:6px;background:#2a3140;color:#d6d9de;cursor:pointer}
.btns button:hover{background:#3a4358} .fb{color:#8fd18f;font-size:12px}
#bar{margin-bottom:12px} #bar button{padding:6px 12px;border-radius:8px;border:0;background:#365; color:#fff;cursor:pointer}
</style>
<h1>imagepipe · session <span id=sid></span></h1>
<div id=bar>
  <button onclick="rerank()">Re-rank with feedback</button>
  <button onclick="exportSel('feedback')">Export by feedback</button>
  <button onclick="exportSel('score')">Export by score</button>
  <button onclick="doReset()" style="background:#8b3a3a; margin-left:16px;">Reset Session</button>
</div>
<div id=searchbar style="margin-bottom: 12px; display: flex; gap: 8px; align-items: center;">
  <select id="provider" style="padding: 6px; border-radius: 8px; border: 1px solid #3a4358; background: #1a1e25; color: white;">
    <option value="openverse">Openverse</option>
    <option value="google_cse">Google CSE</option>
    <option value="local_dir">Local Dir</option>
  </select>
  <input id="q" type="text" placeholder="Enter image prompt..." style="flex-grow: 1; padding: 6px; border-radius: 8px; border: 1px solid #3a4358; background: #1a1e25; color: white;" onkeydown="if(event.key==='Enter') doSearch()">
  <input id="limit" type="number" value="10" min="1" max="100" style="width: 60px; padding: 6px; border-radius: 8px; border: 1px solid #3a4358; background: #1a1e25; color: white;" title="Number of images to load">
  <button onclick="doSearch()" style="padding:6px 12px;border-radius:8px;border:0;background:#365; color:#fff;cursor:pointer">Search</button>
</div>
<div id=status style="margin-bottom: 12px; color: #7fb2d9; display: flex; justify-content: space-between;">
  <span id="statText">Ready.</span>
  <span id="timer" style="color: #9aa1ab; font-family: monospace;"></span>
</div>
<div class=grid id=grid></div>
<script>
async function load(){
  const stat = document.getElementById('statText');
  stat.textContent = 'Loading images...';
  const r = await fetch('/api/images?_t=' + Date.now()); const d = await r.json();
  document.getElementById('sid').textContent = d.session;
  stat.textContent = `Loaded ${d.images.length} images. Ready.`;
  const g = document.getElementById('grid'); g.innerHTML='';
  for(const im of d.images){
    const c = document.createElement('div'); c.className='card';
    c.innerHTML = `<img src="/preview/${im.id}">
      <div class=meta>score <b>${im.rank_score??'-'}</b> · img ${fmt(im.sim_image)} · txt ${fmt(im.sim_text)}<br>
      ${im.width}x${im.height} · ${im.source_domain||'local'} · lic: ${im.license||'?'}
      ${im.is_dup_keeper? '': ' · <i>dup</i>'}<br>
      <a href="${im.source_page_url||'#'}" target=_blank>open source page</a>
      <span class=fb id="fb-${im.id}">${im.fb||''}</span></div>
      <div class=btns>
      ${btn(im.id,'keep','Keep')}${btn(im.id,'reject','Reject')}
      ${btn(im.id,'more_like_this','More like this')}${btn(im.id,'less_like_this','Less')}
      ${btn(im.id,'favorite','★')}${btn(im.id,'uncertain','?')}</div>`;
    g.appendChild(c);
  }
}
const fmt=v=>v==null?'-':(+v).toFixed(2);
const btn=(id,s,t)=>`<button onclick="fb('${id}','${s}')">${t}</button>`;
async function fb(id,s){await fetch(`/api/feedback/${id}/${s}`,{method:'POST'});
  document.getElementById('fb-'+id).textContent=' '+s;}
async function rerank(){
  const stat = document.getElementById('statText');
  const timer = document.getElementById('timer');
  const startTime = Date.now();
  let timerInterval = setInterval(() => {
    timer.textContent = ((Date.now() - startTime)/1000).toFixed(1) + 's';
  }, 100);

  stat.textContent = 'Re-ranking with feedback...';
  let pollInterval = setInterval(async () => {
    try {
      let p = await fetch('/api/progress').then(res => res.json());
      if(p.active) stat.textContent = `Re-ranking: ${p.msg} (${p.pct}%)`;
    } catch(e) {}
  }, 500);

  await fetch('/api/rerank',{method:'POST'});
  clearInterval(pollInterval);
  clearInterval(timerInterval);
  load();
}
async function doSearch(){
  const q = document.getElementById('q').value;
  const limit = document.getElementById('limit').value || 10;
  const provider = document.getElementById('provider').value;
  if(!q) return;
  const stat = document.getElementById('statText');
  const timer = document.getElementById('timer');
  const startTime = Date.now();
  let timerInterval = setInterval(() => {
    timer.textContent = ((Date.now() - startTime)/1000).toFixed(1) + 's';
  }, 100);

  try {
    stat.textContent = 'Step 1/3: Searching provider...';
    let r = await fetch('/api/acquire?query='+encodeURIComponent(q)+'&limit='+encodeURIComponent(limit)+'&provider='+encodeURIComponent(provider), {method:'POST'});
    let res = await r.json();
    if(r.status !== 200) throw new Error(res.error || 'Search failed');
    if(res.results === 0) throw new Error('Provider returned 0 results for this query.');
    
    stat.textContent = 'Step 2/3: Downloading candidates...';
    r = await fetch('/api/download', {method:'POST'});
    res = await r.json();
    if(r.status !== 200) throw new Error(res.error || 'Download failed');

    stat.textContent = 'Step 3/3: Embedding and Ranking...';
    let pollInterval = setInterval(async () => {
      try {
        let p = await fetch('/api/progress').then(res => res.json());
        if(p.active) {
          stat.textContent = `Step 3/3: ${p.msg} (${p.pct}%)`;
        }
      } catch(e) {}
    }, 500);

    r = await fetch('/api/rerank?query='+encodeURIComponent(q), {method:'POST'});
    clearInterval(pollInterval);
    res = await r.json();
    if(r.status !== 200) throw new Error(res.error || 'Rank failed');

    stat.textContent = 'Completed!';
  } catch (e) {
    stat.textContent = 'Error: ' + e.message;
  } finally {
    clearInterval(timerInterval);
    load();
  }
}
async function doReset(){
  if(!confirm('Create a new empty session?')) return;
  const stat = document.getElementById('statText');
  stat.textContent = 'Resetting session...';
  await fetch('/api/reset', {method:'POST'});
  document.getElementById('q').value = '';
  load();
}
async function exportSel(st){
  const stat = document.getElementById('statText');
  stat.textContent = `Exporting by ${st}...`;
  const r=await fetch('/api/export/'+st,{method:'POST'});
  stat.textContent = 'Exported to '+(await r.json()).path;
}
load();
</script>"""


def make_app(cfg: Config, session_id: str) -> FastAPI:
    s = Session(cfg, session_id=session_id)
    app = FastAPI()

    progress_state = {"msg": "", "pct": 0, "active": False}
    def set_progress(msg, pct):
        progress_state["msg"] = msg
        progress_state["pct"] = pct

    @app.get("/api/progress")
    def get_progress():
        return progress_state

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/images")
    def images():
        rows = [dict(r) for r in s.con.execute(
            """SELECT id, rank_score, sim_image, sim_text, width, height, source_domain,
               license, source_page_url, is_dup_keeper FROM images
               WHERE session_id=? AND status='downloaded' AND role='candidate'
               ORDER BY rank_score DESC NULLS LAST""", (s.id,))]
        fb = {r["image_id"]: r["signal"] for r in s.con.execute(
            "SELECT image_id, signal FROM feedback WHERE session_id=? ORDER BY created_at", (s.id,))}
        
        filtered_rows = [r for r in rows if fb.get(r["id"]) != "reject"]
        for r in filtered_rows:
            r["fb"] = fb.get(r["id"])
        return {"session": s.id, "images": filtered_rows}

    @app.get("/preview/{image_id}")
    def preview(image_id: str):
        row = s.con.execute("SELECT preview_path FROM images WHERE id=?", (image_id,)).fetchone()
        if row and row["preview_path"] and Path(row["preview_path"]).exists():
            return FileResponse(row["preview_path"])
        return JSONResponse({"error": "no preview"}, status_code=404)

    @app.post("/api/feedback/{image_id}/{signal}")
    def feedback(image_id: str, signal: str):
        s.feedback(image_id, signal)
        return {"ok": True}

    @app.post("/api/rerank")
    def rerank(query: str = None):
        try:
            progress_state["active"] = True
            progress_state["msg"] = "Preparing embeddings..."
            progress_state["pct"] = 0
            s.embed_all(progress_cb=set_progress)
            set_progress("Ranking candidates...", 100)
            s.rank(text_prompt=query or s.intent or None)
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            progress_state["active"] = False

    @app.post("/api/acquire")
    def acquire(query: str, limit: int = 10, provider: str = "openverse"):
        old_provider = s.cfg.provider
        try:
            s.cfg.provider = provider
            sid = s.search(query, limit=limit)
            count = s.con.execute("SELECT result_count FROM searches WHERE id=?", (sid,)).fetchone()["result_count"]
            return {"ok": True, "results": count}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            s.cfg.provider = old_provider

    @app.post("/api/download")
    def download_api():
        try:
            s.download_pending()
            return {"ok": True}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/reset")
    def reset():
        nonlocal s
        s = Session(cfg, intent=s.intent)
        return {"ok": True, "session": s.id}

    @app.post("/api/export/{strategy}")
    def export(strategy: str):
        return {"path": str(s.export(strategy))}

    return app
