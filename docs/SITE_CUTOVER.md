# Website cutover: v1.0 to v1.1

The submitted site (v1.0) stays reachable at `/v1.0/` on every build. It is a static
build of tag `v1.0-submission` made with `vite build --base=/v1.0/`, committed under `web/public/v1.0/` and never edited.

## State

`main` carries v1.1 and the Netlify production site builds from it. Tag `v1.0-submission` is the submitted code and
`/v1.0/` is the submitted page. The upload ("Try it") section is hidden on hosted builds until `VITE_ZELOS_API` is set
to a running demo server (see [SPACE.md](SPACE.md)); the samples use the template explainer until they are re-exported
with live LLM drafts (`python scripts/export_site_data.py --provider workers_ai`).

## Roll back the site to v1.0

In Netlify, Deploys, publish an earlier deploy of `main`, or temporarily set the production branch to a branch created
from the tag: `git branch v1.0-site v1.0-submission && git push origin v1.0-site`.

## Rebuild the v1.0 snapshot (only if it must be regenerated)

```bash
git archive v1.0-submission web | tar -x -C /tmp/v10
cd /tmp/v10/web && npm ci && npx vite build --base=/v1.0/
rm -rf <repo>/web/public/v1.0 && cp -r dist <repo>/web/public/v1.0
```
