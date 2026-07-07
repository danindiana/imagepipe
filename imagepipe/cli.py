"""CLI entry point: `python -m imagepipe.cli --help`"""
from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .pipeline import Session

app = typer.Typer(help="Image search / similarity / acquisition pipeline (API-compliant).")


@app.command()
def new(intent: str = typer.Option("", help="Natural language goal"),
        config: str = typer.Option(None, "--config", "-c")):
    """Create a new session; prints session id."""
    s = Session(load_config(config), intent=intent)
    typer.echo(s.id)


@app.command()
def add_ref(session: str, paths: list[Path], config: str = typer.Option(None, "-c")):
    """Add reference images (uploads)."""
    s = Session(load_config(config), session_id=session)
    for p in paths:
        typer.echo(f"{p} -> {s.add_reference(p)}")


@app.command()
def crop_screenshot(session: str, screenshot: Path,
                    boxes: str = typer.Option(..., help="x1,y1,x2,y2;x1,y1,x2,y2;..."),
                    config: str = typer.Option(None, "-c")):
    """Extract user-selected regions from a screenshot as query references only."""
    s = Session(load_config(config), session_id=session)
    bx = [tuple(int(v) for v in b.split(",")) for b in boxes.split(";")]
    for iid in s.add_screenshot_crops(screenshot, bx):  # type: ignore[arg-type]
        typer.echo(iid)


@app.command()
def search(session: str, query: str, limit: int = 30, config: str = typer.Option(None, "-c")):
    """Run a provider search and register candidates (no downloads yet)."""
    s = Session(load_config(config), session_id=session)
    typer.echo(s.search(query, limit))


@app.command()
def download(session: str, config: str = typer.Option(None, "-c")):
    """Download all pending candidates through the policy gate, then normalize."""
    Session(load_config(config), session_id=session).download_pending()


@app.command()
def rank(session: str, prompt: str = typer.Option(None, help="Text prompt for hybrid ranking"),
         config: str = typer.Option(None, "-c")):
    """Embed all images, rank candidates, dedup, cluster."""
    s = Session(load_config(config), session_id=session)
    s.embed_all()
    s.rank(text_prompt=prompt)


@app.command()
def label(session: str, config: str = typer.Option(None, "-c")):
    """Zero-shot auto-labeling (requires CLIP backend)."""
    Session(load_config(config), session_id=session).autolabel()


@app.command()
def feedback(session: str, image_id: str,
             signal: str = typer.Argument(..., help="keep|reject|more_like_this|less_like_this|favorite|duplicate|uncertain"),
             config: str = typer.Option(None, "-c")):
    Session(load_config(config), session_id=session).feedback(image_id, signal)


@app.command()
def export(session: str, strategy: str = "score", top: int = 200,
           config: str = typer.Option(None, "-c")):
    """Sort into folders (score|feedback|cluster|domain|license|dedup) + CSV + contact sheet."""
    out = Session(load_config(config), session_id=session).export(strategy, top)
    typer.echo(str(out))


@app.command()
def ui(session: str, config: str = typer.Option(None, "-c")):
    """Launch the local review UI."""
    import uvicorn
    from .webui import make_app
    cfg = load_config(config)
    uvicorn.run(make_app(cfg, session), host=cfg.ui_host, port=cfg.ui_port)


if __name__ == "__main__":
    app()
