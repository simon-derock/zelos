# Zelos demo API for a Hugging Face Space (Docker SDK). Build from the repository root:
#   docker build -f space/Dockerfile -t zelos-space .
# Lightweight runtime only: no torch, no transformers. QC is classical image statistics; retrieval is BM25 plus
# cached embeddings; the explainer is the deterministic template (no LLM key needed).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    ZELOS_ENV=prod \
    ZELOS_PROXY_HOPS=1 \
    ZELOS_RATE_PER_MIN=20

RUN useradd -m -u 1000 user
WORKDIR /app

COPY space/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY --chown=user:user src/ /app/src/
COPY --chown=user:user configs/ /app/configs/
COPY --chown=user:user corpus/ /app/corpus/
# Only the statistics the API reads; copied when present (a missing file falls back to the pre-registered defaults).
COPY results/ /tmp/results-all/
RUN mkdir -p /app/results \
    && for f in reference reference_v11 calibration signatures rule_citations; do \
         if [ -f "/tmp/results-all/$f.json" ]; then cp "/tmp/results-all/$f.json" /app/results/; fi; \
       done \
    && rm -rf /tmp/results-all \
    && chown -R user:user /app

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "zelos.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--no-server-header"]
