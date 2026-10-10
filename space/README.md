---
title: Zelos
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Evidence-gated QC for organ-on-a-chip brightfield images
---

# Zelos demo API

HTTP API behind the "Try your own image" section of the Zelos site
(https://zelos.philipsimonderock.com). It runs the same code as the repository:
physics QC, cause attribution, cited evidence and the grounding gate.

- `GET  /api/health` liveness, version, retriever state
- `POST /api/inspect` QC certificate for one image (multipart field `file`)
- `POST /api/explain?provider=template` gated explanation and claim ledger
- `POST /api/triage` verdicts for 4 to 12 images against a batch reference (multipart field `files`)

Images are processed in memory and not stored. Limits: PNG, JPEG or TIFF, 25 MB and 40 megapixels per image,
20 requests per minute per client.

Research tool. Not a medical device. No clinical or diagnostic claims.
Source: https://github.com/simon-derock/zelos (MIT).
