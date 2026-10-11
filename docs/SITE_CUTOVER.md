# Website cutover: v1.0 to v1.1

The submitted site (v1.0) stays reachable at `/v1.0/` on every build of the `next` branch. It is a static
build of tag `v1.0-submission` made with `vite build --base=/v1.0/`, committed under `web/public/v1.0/` and never edited.

## Switch the main domain to v1.1

1. Hugging Face Space is running and `VITE_ZELOS_API` is set in Netlify (see [SPACE.md](SPACE.md)).
2. Samples re-exported with live LLM drafts: `python scripts/export_site_data.py --provider workers_ai`.
3. Netlify: Site configuration, Build and deploy, Branches: set **Production branch** to `next`.
4. Check `https://zelos.philipsimonderock.com/` (v1.1) and `https://zelos.philipsimonderock.com/v1.0/` (as submitted).

Switch back by setting the production branch to `main`. Nothing in the repository changes.

## Rebuild the v1.0 snapshot (only if it must be regenerated)

```bash
git archive v1.0-submission web | tar -x -C /tmp/v10
cd /tmp/v10/web && npm ci && npx vite build --base=/v1.0/
rm -rf <repo>/web/public/v1.0 && cp -r dist <repo>/web/public/v1.0
```
