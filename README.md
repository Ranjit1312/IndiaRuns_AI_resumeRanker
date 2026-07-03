---
title: Resume-Fit — JD Compiler + Résumé Fit
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Resume-Fit — JD → profile compiler + Résumé → fit scorer

Built on the [RedRob interpretable ranking engine](https://github.com/Ranjit1312/redrob_indiaRuns).
Two flows in one Streamlit app:

1. **JD → profile (Module 1):** turn any job description into a **schema-valid
   `jd_profile.yaml`** (the config that drives the ranking engine) + a coaching
   `jd_meta.yaml`. The engineering is in the harness, not the LLM call:
   schema-constrained generation → validation against `jd/jd_profile.schema.json`
   → a **bounded self-repair loop** → **per-section decomposition** fallback →
   sentinel/default graceful degradation. Never emits an invalid file.
2. **Résumé → fit (Module 2):** upload/paste a résumé → RLM harness parses it into
   the candidate schema → an editable **HITL** form → **interpretable fit
   scorecard + per-signal gap analysis** (rules engine, no LightGBM). Supports
   **batch** résumés (a ranked leaderboard) and a **Dashboard** that runs a live
   A/B/C comparison study.

- **Provider:** Google AI Studio, **Gemma 4** (`gemma-4-26b-a4b-it` default,
  `gemma-4-31b-it`; `gemini-3-flash` selectable). **Bring your own key** — pasted
  in the sidebar, held only for your session, never stored or logged. Résumé
  embeddings use `gemini-embedding` on the same key.
- **Validation is authoritative:** reuses the engine's own
  `redrob_ranker.profile.load` (JSON-Schema + regex + precise field errors).
- **Resilience:** transient Google `500`/`503` errors are retried with backoff
  (distinct from the `429` rate-cap path); live per-leaf telemetry (prompt/tokens/
  latency) is shown in-app. Image-only/scanned PDFs are OCR'd via Gemma multimodal
  (5 passes/page) with a once-only Gemini fallback — **no GPU required**.

---

## Run locally

> **Why local?** The synthetic évaluation résumés (`data/synthetic_resumes/`) are
> **git-ignored** — the image-only PDFs are ~12 MB each and exceed the HF Space
> 10 MB/file limit. So the **Dashboard "Run live comparison"** button and the
> résumé eval set only work on a local clone, where you generate them first
> (below). The hosted HF Space runs the JD + résumé flows; the comparison study
> is a local-only feature.

### 1. Clone + set up
```bash
git clone https://github.com/Ranjit1312/IndiaRuns_AI_resumeRanker.git
cd IndiaRuns_AI_resumeRanker

python -m venv .venv
# Windows:
./.venv/Scripts/python -m pip install -r requirements.txt
# macOS/Linux:  ./.venv/bin/python -m pip install -r requirements.txt

# optional — only for the PDF/scanned-résumé variants + local DuckDB/LanceDB:
./.venv/Scripts/python -m pip install -r requirements-db.txt
```

### 2. Generate the evaluation set (local only)
This writes the labeled synthetic résumés the comparison study reads:
```bash
./.venv/Scripts/python -m tools.gen_resumes
# → data/synthetic_resumes/<jd_slug>_<tier>/<layout>/resume.(txt|docx|pdf)
#   (single_column/two_column/table_heavy/ats_plain/image_only)
```

### 3. Run the app
```bash
./.venv/Scripts/python -m streamlit run app.py
```
Open the printed URL and paste your **Google AI Studio API key** in the sidebar
(free at aistudio.google.com; held for the session only).

### 4. Use the live-comparison button
In the app: **Dashboard** tab → set the number of (JD, résumé) pairs + repeats →
**"Run live comparison"**. It runs our engine vs. two naive single-LLM baselines
over the labeled set and writes:
- `data/eval_study/comparison_report.md` + `scores.csv` (latest), and
- a timestamped copy under `data/eval_study/runs/<timestamp>/`.

The Dashboard then renders ranking-accuracy (vs. the known strong/borderline/weak
tiers), score stability across repeats, and token/latency cost per method. These
are computed on **synthetic labeled data** — a sanity check of ranking agreement,
**not** a live percentile.

> Prefer the CLI? Same study headless:
> ```bash
> ./.venv/Scripts/python -m eval_study.compare --key <KEY> --models "gemma-4-26b-a4b-it" --repeats 3
> # offline sanity check (no key, mock backend):
> ./.venv/Scripts/python -m eval_study.compare --mock
> ```

## Tests
```bash
./.venv/Scripts/python -m pytest tests/ -q
```

## Layout
```
app.py                       # Streamlit UI: JD→profile · Résumé→fit (+batch) · Dashboard
harness/                     # backends, prompts, coerce (repair+decompose), validate,
                             #   ingest (shared file→text + OCR), resume, logging/telemetry
redrob_ranker/               # profile.py (authoritative validator), intrinsic.py,
                             #   fit.py (single-candidate interpretable scorer)
store/                       # session-safe MemoryStore default; DuckDB/LanceDB behind a
                             #   Workspace facade (swappable to enterprise DBs)
eval_study/compare.py        # A/B/C comparison study (our engine vs single-LLM baselines)
tools/gen_resumes.py         # synthetic labeled résumé generator (git-ignored output)
jd/                          # schema, method_config, golden profile (retained from Repo 1)
data/eval_jds/               # gold JD eval set (profile+meta, validated)
candidate_schema.json        # candidate profile schema (retained from Repo 1)
```

## Roadmap
- **Done:** Module 1 (JD→profile), Module 2 (résumé→fit + gap analysis, batch,
  Dashboard), embedded-DB seam, OCR, comparison study.
- **Next:** local Ollama tier (Gemma 4 E4B + EmbeddingGemma) for an offline,
  zero-egress posture; enterprise DB backends via the same `store` facade.
