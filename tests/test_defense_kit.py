"""The defense kit is generated from results/*.json: no placeholder survives, no figure is typed by hand."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SITE = ROOT / "web" / "public" / "data" / "site.json"
THUMBS = ROOT / "web" / "public" / "data" / "thumbs"

needs_inputs = pytest.mark.skipif(
    not (RES / "results.json").exists() or not SITE.exists() or not THUMBS.exists(),
    reason="results, site.json or thumbnails missing",
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_defense_kit", ROOT / "scripts" / "build_defense_kit.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["build_defense_kit"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    gen = _load_generator()
    out = tmp_path_factory.mktemp("defense")
    written = gen.build(out, RES, SITE, THUMBS, write_snapshot=False)
    return gen, out, {rel: p.read_text() for rel, p in written.items()}


# ----------------------------------------------------------------------------- provenance helpers
def _floats(obj, acc: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _floats(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _floats(v, acc)


def _allowed_strings(gen, results_dir: Path) -> set[str]:
    """Every way a number in results/*.json can be written: 1 to 4 decimals, as a fraction or a percentage."""
    vals: set[float] = set()
    for f in results_dir.glob("*.json"):
        try:
            _floats(json.loads(f.read_text()), vals)
        except ValueError:
            continue
    v1 = gen.load_v1()
    _floats({k: v for k, v in v1.items() if k not in ("numbers", "source")}, vals)
    out: set[str] = set()
    for x in vals:
        for d in range(1, 5):
            out.add(f"{x:.{d}f}")
            out.add(f"{100 * x:.{d}f}%")
            out.add(f"{x:.{d}f}%")  # values already stored as percentages
    # per-image QC scores on the sample certificates are real pipeline output exported to site.json
    if SITE.exists():
        site: set[float] = set()
        _floats(json.loads(SITE.read_text())["samples"], site)
        for x in site:
            for d in range(1, 5):
                out.add(f"{x:.{d}f}")
    out |= gen.numbers_in((results_dir / "results.md").read_text())
    out |= set(v1.get("numbers", []))
    return out


def _visible_text(name: str, text: str) -> str:
    if name.endswith(".html"):
        text = re.sub(r"<style.*?</style>|<script.*?</script>", " ", text, flags=re.S)
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)  # tags and their attributes (SVG coordinates, data URIs)
    else:
        text = re.sub(r"<!--np-->.*?<!--/np-->", "", text, flags=re.S)  # same escape hatch as the README
        text = re.sub(r"https?://\S+|\]\([^)]*\)", "", text)
        text = re.sub(r"`[^`]*`", " ", text)  # file paths and field names
    return text


# ----------------------------------------------------------------------------- tests
@needs_inputs
def test_no_placeholder_survives(kit):
    _, _, files = kit
    assert set(files) == {"docs/defense/index.html", "docs/DEFENSE_QA.md", "docs/DEFENSE_TALK_TRACK.md",
                          "docs/V1_1_CHANGES.md"}
    for name, text in files.items():
        assert "{{" not in text and "}}" not in text, name
        assert len(text) > 2000, name


@needs_inputs
def test_every_decimal_and_percentage_comes_from_results(kit):
    """Strong check: each figure was produced by a formatting helper from a results file (a number typed into a
    template or into the generator is not in EMITTED). Weak check, as in evaluate.check_provenance: it also
    appears in results.md or equals a value in results/*.json."""
    gen, _, files = kit
    allowed = _allowed_strings(gen, RES)
    for name, text in files.items():
        found = gen.numbers_in(_visible_text(name, text))
        assert found, f"{name}: no figures at all, templates were not filled"
        typed = sorted(n for n in found if n not in gen.EMITTED)
        assert not typed, f"{name}: figures not produced by a helper from the results: {typed}"
        missing = sorted(n for n in found if n not in allowed)
        assert not missing, f"{name}: numbers not found in results: {missing}"


@needs_inputs
def test_a_typed_number_would_be_caught(kit):
    gen, _, files = kit
    sneaky = files["docs/DEFENSE_QA.md"] + "\nWe reach 99.87% on everything."
    found = gen.numbers_in(_visible_text("x.md", sneaky))
    assert "99.87%" in found and "99.87%" not in gen.EMITTED


@needs_inputs
def test_unknown_placeholder_is_an_error(kit):
    gen, _, _ = kit
    with pytest.raises(KeyError):
        gen.render("value: {{does.not.exist}}", {"a": "1"}, is_html=False)


@needs_inputs
def test_deck_is_self_contained_and_complete(kit):
    _, _, files = kit
    html = files["docs/defense/index.html"]
    assert len(re.findall(r'<section class="slide"', html)) == 15
    assert len(re.findall(r'<aside class="notes">', html)) == 15
    assert not re.search(r"<link\b|<script[^>]+src=|@import|url\(\s*['\"]?https?:|(?:src|href)=\"https?:", html)
    assert "prefers-color-scheme: dark" in html and "@media print" in html and "?print" not in html.split("<script>")[0]
    uris = re.findall(r'src="(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+)"', html)
    assert len(uris) == 2
    assert all(len(u) * 3 / 4 <= 600_000 for u in uris)
    for notes in re.findall(r'<aside class="notes">(.*?)</aside>', html, flags=re.S):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", notes.strip()) if s]
        assert 2 <= len(sentences) <= 4, notes[:60]
    for key in ("ArrowRight", "ArrowLeft", "' '", "'n'", "print"):
        assert key in html


@needs_inputs
def test_qa_has_25_questions_with_sources(kit):
    _, _, files = kit
    qa = files["docs/DEFENSE_QA.md"]
    qs = re.findall(r"^### Q(\d+)\. ", qa, flags=re.M)
    assert [int(q) for q in qs] == list(range(1, 26))
    assert qa.count("**Source:**") == 25
    must = ("fine-tune", "expert-label AUROC", "biologically", "circular", "label-trained head do so badly", "grounding gate",
            "literature support", "Cohere", "licensing", "regulatory", "v1.0 and v1.1", "480", "session", "vision-LLM",
            "latency", "upload demo", "pipeline", "sponsor")
    low = qa.lower()
    for m in must:
        assert m.lower() in low, m


@needs_inputs
def test_honesty_and_attribution_rules(kit):
    _, _, files = kit
    gen_src = (ROOT / "scripts" / "build_defense_kit.py").read_text()
    templates = [p.read_text() for p in (ROOT / "docs" / "defense" / "templates").glob("*.tmpl")]
    banned = re.compile(r"claude|anthropic|ai assistant|co-authored", re.I)
    for name, text in {**files, "generator": gen_src, **{f"tmpl{i}": t for i, t in enumerate(templates)}}.items():
        assert not banned.search(text), name
    qa = files["docs/DEFENSE_QA.md"]
    low = qa.lower()
    assert "do not claim to beat" in low or "do not claim state-of-the-art" in low
    changes = files["docs/V1_1_CHANGES.md"]
    assert "post-deadline" in changes and "v1.0-submission" in changes
    for key in ("single-image absolute mode", "n=", "motion blur", "expert-labelled neighbours"):
        assert key.lower() in changes.lower(), key
    # every limitation row in docs/LIMITATIONS.md has a counterpart on the limitations slide or in the changelog
    html = files["docs/defense/index.html"].lower()
    for key in ("cross-session", "attribution", "expert label", "single-lab", "gold set", "lexical", "small",
                "content hashes", "clinical"):
        assert key in html, key
