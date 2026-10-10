"""Zelos command line: `zelos inspect|explain|triage|ask|mcp`.

inspect <image>            QC certificate JSON (metrics, level, causes, actions); 0 LLM calls
explain <image>            gated explanation + claim ledger (provider: workers_ai | template)
triage <folder>            session triage: every image QC'd against the folder's own batch reference
ask "<question>"           cited literature answer, or ABSTAIN
mcp                        start the MCP server (stdio)
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Zelos: evidence-gated QC for organ-on-a-chip images (research use only).")


def _retriever():
    try:
        from zelos.rag.retriever import HybridRetriever

        return HybridRetriever.load()
    except Exception:
        return None


@app.command()
def inspect(image: Path, cell_line: str = typer.Option(None)) -> None:
    from zelos.vision.pipeline import inspect as qc

    typer.echo(json.dumps(qc(image, cell_line=cell_line), indent=2))


@app.command()
def explain(image: Path, provider: str = "template", cell_line: str = typer.Option(None)) -> None:
    from zelos.agent.loop import run_inspect
    from zelos.vision.pipeline import inspect as qc

    out = run_inspect(qc(image, cell_line=cell_line), retriever=_retriever(), provider=provider)
    typer.echo(json.dumps({k: out.get(k) for k in ("card", "ledger", "usage", "provider")}, indent=2))


@app.command()
def triage(folder: Path, provider: str = "template") -> None:
    import pandas as pd

    from zelos.agent.loop import run_triage
    from zelos.vision.metrics import BAD_DIRECTION, compute_metrics
    from zelos.vision.pipeline import inspect as qc
    from zelos.vision.pipeline import read_image
    from zelos.vision.scoring import fit_reference

    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"})
    if not paths:
        raise typer.BadParameter("no images found")
    ref = fit_reference(pd.DataFrame([compute_metrics(read_image(p)[0]) for p in paths])[list(BAD_DIRECTION)])
    qcs = [qc(p, reference=ref) for p in paths]
    out = run_triage(qcs, retriever=_retriever(), provider=provider)
    typer.echo(json.dumps({k: out.get(k) for k in ("card", "ledger", "usage")}, indent=2, default=str))


@app.command()
def ask(question: str, provider: str = "template") -> None:
    from zelos.agent.loop import run_ask

    out = run_ask(question, retriever=_retriever(), provider=provider)
    typer.echo(json.dumps({k: out.get(k) for k in ("card", "ledger", "usage")}, indent=2))


@app.command()
def mcp() -> None:
    from zelos.mcp_server import mcp as server

    server.run()


if __name__ == "__main__":
    app()
