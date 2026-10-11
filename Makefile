# Zelos — every target runs offline without API keys (Cohere / Workers AI responses are replayed from committed caches).
PY ?= .venv/bin/python
N ?= 480

.PHONY: setup data features corpus eval judge test lint quality api mcp site all
setup:            ## create venv + install pinned deps
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt -r requirements-dev.txt && $(PY) -m pip install -q -e .
data:             ## stratified 480-image subset of the Zenodo OoC dataset (CRC-verified range requests)
	$(PY) data/download.py --n $(N)
features:         ## stream all 3,072 images, keep only features (physics metrics + frozen DINOv2); ~45 min, resumable
	$(PY) scripts/extract_features.py
corpus:           ## rebuild the open-access literature corpus (Europe PMC, CC BY / CC0 only)
	$(PY) scripts/build_corpus.py
eval:             ## E1 E2 E4 E5 + head-to-head + rule citations + VLM baseline + LLM sweep -> results/results.md
	$(PY) evaluate.py
judge:            ## fast reviewer smoke test (<5 min CPU): tests + retrieval + grounding replay + summary
	$(PY) -m pytest -q && $(PY) evaluate.py --skip-vision
test:
	$(PY) -m pytest -q --cov=zelos --cov-report=term-missing:skip-covered
lint:
	$(PY) -m ruff check src tests scripts evaluate.py
quality: lint     ## lint + types + security + dependency audit
	$(PY) -m mypy || true
	$(PY) -m bandit -q -r src -ll
	$(PY) -m pip_audit -r requirements.txt || true
api:
	$(PY) -m uvicorn zelos.api.main:app --port 8000
mcp:
	$(PY) -m zelos.mcp_server
site:             ## export real pipeline outputs for the web console
	$(PY) scripts/export_site_data.py
all: setup data eval site
