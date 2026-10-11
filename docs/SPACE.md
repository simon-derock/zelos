# Live demo server: Hugging Face Space

The site section "Try your own image" talks to the Zelos API. The API runs as a Docker Space on the free
CPU tier (the lightweight runtime only: no torch, no transformers, no API keys needed). The site finds it
through one build-time variable, `VITE_ZELOS_API`. When the variable is not set the section shows a
"not configured" panel with the local commands instead of an upload box.

## One-time setup (5 steps)

1. **Create the Space.** Open <https://huggingface.co/new-space>. Name it `zelos`, choose SDK **Docker**
   (Blank), hardware **CPU basic (free)**, visibility **Public**. Or from a terminal:
   `huggingface-cli repo create zelos --type space --space_sdk docker`.
2. **Create a write token.** <https://huggingface.co/settings/tokens> > New token > role **Write**. Copy it.
3. **Add two GitHub secrets.** Repository > Settings > Secrets and variables > Actions > New repository secret:
   - `HF_TOKEN` = the token from step 2
   - `HF_SPACE` = `<your-hf-username>/zelos`
4. **Deploy.** Push to branch `main` or `next` (any change under `src/`, `configs/`, `corpus/` or `space/`), or run
   Actions > Deploy Space > Run workflow. The first build takes 2 to 4 minutes. Check
   `https://<your-hf-username>-zelos.hf.space/api/health`: it should answer `{"status":"ok", ...}`.
5. **Point the site at it.** Netlify > Site configuration > Environment variables > Add a variable:
   `VITE_ZELOS_API` = `https://<your-hf-username>-zelos.hf.space` (no trailing slash). Then
   Deploys > Trigger deploy > **Clear cache and deploy site** (Vite reads the variable at build time).

Both Netlify configs already allow `https://*.hf.space` in the `connect-src` of the Content Security Policy, and
the API allows the two production origins for CORS.

## Deploy without GitHub Actions

```sh
sh space/stage.sh /tmp/zelos-space          # run from the repository root
cd /tmp/zelos-space && git init -q -b main && git add -A && git commit -qm deploy
git push --force https://<user>:<HF_TOKEN>@huggingface.co/spaces/<user>/zelos main
```

## Local run

```sh
make api                                   # http://localhost:8000
cd web && VITE_ZELOS_API=http://localhost:8000 npm run dev
```

Or the exact container the Space runs:

```sh
docker build -f space/Dockerfile -t zelos-space .
docker run --rm -p 7860:7860 zelos-space   # http://localhost:7860/api/health
```

## What the server does and does not do

- Accepts PNG, JPEG or TIFF, at most 25 MB and 40 megapixels per image; the header is checked before decoding.
- Images are processed in memory and are never written to disk or logged.
- Rate limit: 20 requests per minute per client (a session triage costs one per image), 429 with `Retry-After`.
- The explainer is the deterministic template, so no language-model key is needed and the Space makes no
  outbound calls. Retrieval uses the committed corpus and cached embeddings.
- Free Spaces sleep when idle; the first request after a pause can take about a minute.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZELOS_ENV` | `prod` | `dev` allows any CORS origin and skips retriever preloading |
| `ZELOS_CORS_ORIGINS` | the two production origins | comma-separated allow-list (localhost origins are always allowed) |
| `ZELOS_RATE_PER_MIN` | `20` | per-client requests per minute |
| `ZELOS_RATE_GLOBAL_PER_MIN` | 10 x per-client | ceiling for all clients together |
| `ZELOS_PROXY_HOPS` | `1` in prod, `0` in dev | trusted reverse proxies in front of the app (client address comes from `X-Forwarded-For`) |
| `ZELOS_MAX_CONCURRENCY` | `2` | images analysed at once |

Research tool. Not a medical device. No clinical or diagnostic claims.
