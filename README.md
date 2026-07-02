---
title: Resume-Fit — JD Compiler
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Resume-Fit — JD → profile compiler (v1, Module 1)

A **model-agnostic harness** that turns any job description into a **schema-valid
`jd_profile.yaml`** (the config that drives the [RedRob interpretable ranking
engine](https://github.com/Ranjit1312/redrob_indiaRuns)) plus a coaching sidecar
`jd_meta.yaml`. The engineering is in the harness, not the LLM call:
schema-constrained generation → validation against `jd/jd_profile.schema.json` →
a **bounded self-repair loop** → **granular per-section decomposition** fallback →
sentinel/default graceful degradation. Never emits an invalid file.

- **Provider:** Google AI Studio, **Gemma 4** (`gemma-4-26b-a4b-it` default,
  `gemma-4-31b-it`; `gemini-3-flash` selectable). **Bring your own key** — pasted
  in the sidebar, held only for your session, never stored server-side or logged.
  Free within Google AI Studio limits, subject to their rate caps.
- **Validation is authoritative:** reuses the engine's own
  `redrob_ranker.profile.load` (JSON-Schema + regex-compilation + precise field
  errors).

## Roadmap
- **v1 (this):** Module 1 — JD → `jd_profile.yaml` + `jd_meta.yaml`, parity eval,
  HF Space. Hosted Gemma 4.
- **Phase 2:** Module 2 — resume → candidate schema → **rules-engine fit + per-signal
  gap analysis** (interpretable, no LGBM; `gemini-embedding`); local Ollama tier
  (Gemma 4 E4B + EmbeddingGemma); no-backend/client-side posture.

## Local run
```bash
python -m venv .venv && ./.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY (or paste it in the UI)
./.venv/Scripts/python -m streamlit run app.py
```

## Layout
```
app.py                     # Streamlit UI (BYO key, compile, HITL edit, parity)
harness/                   # backends, prompts, coerce (repair+decompose), validate, parity
jd/                        # retained from Repo 1: schema, method_config, golden profile, RETARGETING
redrob_ranker/profile.py   # retained: the authoritative validator (loaded standalone)
data/eval_jds/             # gold eval set: 30–50 cross-domain real JDs (profile+meta, validated)
```
