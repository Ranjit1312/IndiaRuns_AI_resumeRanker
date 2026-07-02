# Submission Deck — Paste-Ready Points

Project: **Resume-Fit — JD → profile compiler (v1, Module 1)**, built on the
existing **RedRob interpretable ranking engine** (Repo 1).

> Slide headings below are copied **verbatim** from the challenge Google Slides
> template. Bullets under each are paste-ready and honor the no-overclaiming
> rules: **no benchmark numbers are stated as proven**; any improvement is marked
> **estimated/projected**, and comparisons point to the Repo 1 README / HF demo
> homepage rather than quoting a figure here.

---

## Team Name :
- `<<TEAM_NAME>>`  *(TO FILL)*

## Problem Statement :
- Traditional candidate matching is keyword/ATS-based: it misses context, can't
  explain *why* a candidate ranks where they do, and treats every job description
  the same. Recruiters get opaque scores; candidates get no actionable signal.
- Before any ranking can be trustworthy, the **job description itself must be
  turned into a precise, machine-checkable specification** of what the role
  actually needs — today that step is manual, inconsistent, and error-prone.

## Team Leader Name :
- `<<TEAM_LEADER_NAME>>`  *(TO FILL)*

---

## Solution Overview

**What is your proposed solution?**
- **Resume-Fit** — a candidate-centric tool built on top of the existing
  **RedRob interpretable ranking engine**. This submission delivers **Module 1**:
  a **model-agnostic harness** that compiles *any* job description (free prose)
  into a **schema-valid `jd_profile.yaml`** — the config that drives the ranking
  engine — plus a coaching sidecar `jd_meta.yaml`.
- The compiler is a **depth-1 Recursive Language Model (RLM) harness**
  (arXiv 2512.24601): a deterministic root slices the JD and dispatches focused
  leaf extractions (one sub-field each), rather than one giant LLM call.
- Ships on **Google AI Studio — Gemma 4** (BYO API key, session-only, never
  stored), deployed as a **Streamlit Hugging Face Space**. Backend is
  model-agnostic (Gemma 4 variants and Gemini-3-Flash are selectable).

**What differentiates your approach from traditional candidate matching systems?**
- **The engineering is in the harness, not the LLM call.** Schema-constrained
  generation → validation against the engine's own schema → a **bounded
  self-repair loop** (re-call only the failing block) → **per-section
  decomposition** fallback → sentinel/default graceful degradation. Result:
  **it never emits an invalid file.**
- **Validation is authoritative:** we reuse the ranking engine's *own*
  `redrob_ranker.profile.load` (JSON-Schema + regex compilation + precise field
  errors) — so a "valid" profile is valid *by the engine's definition*, not the
  LLM's opinion.
- **RLM framing:** treat the JD as an external variable; a small model performs
  better by learning *when/how to delegate* focused extractions, not by doing one
  monolithic call.
- We validated the compiler against **10 hand-authored gold JDs** across domains
  (Amazon, Anthropic, Databricks, Google, NVIDIA, Stripe).

---

## JD Understanding & Candidate Evaluation

**What are the key requirements extracted from the JD?**
- The harness extracts the JD into the engine's structured schema, including:
  role/domain terms and **`domain.out_of_domain_terms`** (the general
  disqualifier lever), skill/experience signals, and role-behavior flags such as
  `only_consulting`, `stale_ic_role`, and `cv_primary`.
- Hard requirements the engine can't numerically score (degree, travel, visa)
  are captured in the `jd_meta.yaml` sidecar as advisory (`enforced_by: none`),
  so nothing is silently dropped.
- Validated contract findings from the 10 real roles, e.g.: `only_consulting`
  must be **off** for pre-sales / Solutions-Architect / Forward-Deployed roles;
  `stale_ic_role` is per-role (on only for hands-on IC coding); `cv_primary` is
  off for NVIDIA AI/ML (computer-vision background is welcome there).

**Which candidate signals are most important / How does your solution evaluate
candidate fit beyond keyword matching?**
- Fit is driven by the **RedRob interpretable ranking engine**, which scores
  structured per-signal criteria defined by the compiled `jd_profile.yaml` —
  not raw keyword overlap. Signals combine domain relevance, disqualifier terms,
  and role-behavior flags.
- **Module 2 (Phase 2, in progress):** resume → candidate schema → interpretable
  **rules engine** → **per-signal gap analysis** + human-in-the-loop correction,
  so a candidate sees *which* signals helped or hurt and can correct bad parses.
- *Note: the deep candidate-fit scoring is the ranking engine's job — see the
  Repo 1 README for its methodology and any reported comparison numbers.*

---

## Ranking Methodology

**How does your system retrieve, score, and rank candidates?**
- Ranking is performed by the **RedRob interpretable ranking engine** (Repo 1),
  configured by the **`jd_profile.yaml`** this harness produces. This submission
  (Module 1) is the **JD → profile compiler** that makes that ranking precise and
  reproducible.

**What models, algorithms, or heuristics are used?**
- **Compiler:** depth-1 **RLM harness** over **Gemma 4** (model-agnostic;
  Gemini-3-Flash selectable). Deterministic root decomposition + focused leaf
  extractions + bounded repair.
- **Ranking engine:** interpretable, rule/criteria-based scoring driven by the
  schema — **no LightGBM / no black-box model** (interpretability by design).
  See the Repo 1 README for the exact scoring method.

**How are multiple candidate signals combined into a final ranking?**
- Signals are combined by the engine's transparent per-criterion scoring as
  specified in the compiled profile (domain terms, disqualifiers, role-behavior
  flags). Because the profile is schema-validated by the engine's own loader, the
  combination logic is deterministic and auditable.
- *For how the combined ranking compares to direct-LLM-call ranking and other
  methods: **actual comparison numbers are provided in the Repo 1 README and on
  the HF demo app's homepage for reference.***

---

## Explainability & Data Validation

**How are ranking decisions explained?**
- The ranking engine is **interpretable by design** (rules/criteria, no black-box
  model), so each signal's contribution is inspectable. The coaching
  `jd_meta.yaml` sidecar further surfaces *why* each JD requirement matters.
- Module 2 adds **per-signal gap analysis** with a human-in-the-loop correction
  form — every score maps back to a named, editable signal.

**How do you prevent hallucinations or unsupported justifications?**
- **Schema-constrained generation + authoritative validation:** every field must
  pass the engine's own `redrob_ranker.profile.load` (JSON-Schema + regex). An
  LLM cannot introduce a field or value the schema doesn't allow.
- **Bounded self-repair:** on a validation failure the harness re-calls **only
  the failing block**, then falls back to **per-section decomposition**, then to
  **sentinel/default graceful degradation** — so output is always schema-valid,
  never a hallucinated free-text blob. **The harness never emits an invalid file.**

**How does your solution handle inconsistent, low-quality, or suspicious profiles?**
- The same validate → bounded-repair → decompose → sentinel-degrade pipeline
  makes malformed or ambiguous JD input **degrade gracefully to a valid, minimal
  profile** rather than failing or fabricating.
- Gold JDs were fetched **verbatim via structured sources** (Greenhouse API,
  Workday CXS, Amazon JSON, JSON-LD) — never through a summarizer — so eval data
  integrity is preserved.
- *(Candidate-profile anti-gaming / suspicious-resume handling is part of the
  Module 2 rules-engine + HITL work, in progress for Phase 2.)*

---

## End-to-End Workflow

**What is the complete workflow from JD input to ranked candidate output?**
1. **Paste JD** (free prose) into the Streamlit app; paste your own Google AI
   Studio key (session-only).
2. **RLM compile:** deterministic root slices the JD → focused leaf extractions
   fill each schema sub-field.
3. **Validate** against the engine's own loader (`redrob_ranker.profile.load`).
4. **Bounded repair** on any failure (re-call failing block → per-section
   decompose → sentinel degrade). **Guaranteed schema-valid output.**
5. **Emit** `jd_profile.yaml` (drives the ranker) + `jd_meta.yaml` (coaching).
6. **Human-in-the-loop edit** in the UI, then hand off to the **RedRob ranking
   engine**, which scores/ranks candidates against the compiled profile.
7. *(Phase 2)* Resume → candidate schema → rules-engine fit + per-signal gap
   analysis returned to the candidate.

---

## System Architecture

- **UI:** Streamlit app (`app.py`) — BYO-key input, compile, HITL edit, parity
  tab. Deployed as a **Hugging Face Space** (CPU basic, no torch).
- **Harness (`harness/`):** model-agnostic backends, prompts, `coerce`
  (repair + decompose), validate, parity eval.
- **Schema + validator (`jd/`, `redrob_ranker/profile.py`):** retained from
  Repo 1 — the authoritative JSON-Schema and the standalone validator loaded by
  the harness.
- **Models:** Google AI Studio — Gemma 4 (`gemma-4-26b-a4b-it` default,
  `gemma-4-31b-it`; `gemini-3-flash` selectable). **BYO key, session-only.**
- **Data:** `data/eval_jds/` — 10 validated cross-domain gold JDs
  (profile + meta + verbatim source).
- *(Phase 2 planned):* modular embedded DB (**DuckDB + LanceDB**) for
  session/profile records, swappable to enterprise DBs; optional local Ollama
  tier (Gemma 4 E4B + EmbeddingGemma) for a no-backend / client-side posture.

*(Suggest pasting the architecture diagram / flow here on the slide.)*

---

## Results & Performance

**What results or insights demonstrate ranking quality?**
- **Robustness result (qualitative, verified in-repo):** across **10 hand-authored
  gold JDs** spanning security, TPM, PM, pre-sales/SA, FDE, data-science and
  backend roles, the harness produces **schema-valid `jd_profile.yaml` for every
  input** (multi-model parity eval across Gemma 4 variants). **It never emits an
  invalid file.**
- **No accuracy/performance number is claimed as proven here — nothing in this
  repo has been formally benchmarked yet.**
- For quantitative comparison of this approach vs **direct LLM calls for ranking**
  (and vs other methods): ***actual comparison numbers are provided in the Repo 1
  README and on the HF demo app's homepage for reference.***
- Where an improvement is cited, treat it as **estimated/projected** — e.g.
  *"estimated ~Nx improvement in [metric]"* is a **projection, not a measured
  result** (fill `N`/metric only from a cited source; otherwise leave as a range).

**How does your solution meet the challenge's runtime and compute constraints?**
- **Lightweight footprint:** the Space builds from a **torch-free**
  `requirements.txt` and runs on **CPU basic (free)** — no GPU required.
- **Small-model-first:** the RLM design lets a **small hosted model (Gemma 4)**
  do focused sub-extractions instead of one large call, keeping cost/latency low
  and staying within Google AI Studio free-tier rate caps.
- **BYO-key, session-only** compute — no server-side model hosting cost, no stored
  keys or data.

---

## Technologies Used

**What technologies, frameworks, and tools were used and why?**
- **Google AI Studio — Gemma 4** (`gemma-4-26b-a4b-it` / `gemma-4-31b-it`;
  `gemini-3-flash` selectable): free within AI-Studio limits, strong small-model
  extraction, **BYO-key** so no shared cost/secret exposure.
- **Recursive Language Model (RLM) harness pattern** (arXiv 2512.24601): treat
  the JD as an external variable; delegate focused sub-field extractions —
  chosen for reliability and small-model efficiency over a single giant call.
- **JSON-Schema validation** via the engine's own `redrob_ranker.profile.load`:
  makes correctness authoritative and reuses Repo 1 instead of re-implementing.
- **Streamlit + Hugging Face Spaces:** fast interactive UI, free CPU hosting,
  trivial deploy (`sdk: streamlit`, `app_file: app.py`).
- **Structured JD fetchers** (Greenhouse API, Workday CXS, Amazon JSON, JSON-LD):
  verbatim JD capture for the gold eval set — never a summarizer.
- **(Phase 2 planned):** DuckDB + LanceDB (embedded, swappable), Ollama +
  Gemma 4 E4B / EmbeddingGemma for a local/offline tier.

---

## Submission Assets

- **GitHub repo:** `<<GITHUB_REPO_URL>>`
  *(git remote `origin` currently points to
  `https://github.com/Ranjit1312/IndiaRuns_AI_resumeRanker` — confirm it is
  pushed and public before submitting.)*
- **Hugging Face Space (live demo):** `<<HF_SPACE_URL>>`
  *(git remote `space` currently points to
  `https://huggingface.co/spaces/Ranjit1312/Resume_Ranker` — confirm the Space
  is built and public before submitting.)*
- **Demo video:** `<<DEMO_VIDEO_URL>>`  *(TO FILL)*
- **Reference for comparison numbers:** Repo 1 README + HF demo homepage
  (`https://github.com/Ranjit1312/redrob_indiaRuns`).

---

# (a) Placeholders to fill before submitting
- `<<TEAM_NAME>>`, `<<TEAM_LEADER_NAME>>`
- `<<GITHUB_REPO_URL>>` — likely `https://github.com/Ranjit1312/IndiaRuns_AI_resumeRanker`
  (git remote `origin`), **but confirm it is actually pushed + public.**
- `<<HF_SPACE_URL>>` — likely `https://huggingface.co/spaces/Ranjit1312/Resume_Ranker`
  (git remote `space`), **but confirm the Space is built + public.**
- `<<DEMO_VIDEO_URL>>` — no video link found in the repo.
- **Any performance / accuracy / "~Nx improvement" number** — do NOT hardcode.
  Pull only from the **Repo 1 README** or the **HF demo homepage**, and label as
  **estimated/projected** unless the source states it as measured.

# (b) Unverified / could not confirm
- **Deck read succeeded** via the public txt-export endpoint (deck is at least
  link-readable). Headings above are verbatim from that export; only the
  **"System Architecture"** slide had **no question prompt text** in the export
  (title only) — confirm whether that slide expects a diagram vs. written answer.
- **Repo URL / HF Space URL are inferred from git remotes**, not confirmed as
  pushed/live/public. DEPLOY.md still uses `<USER>/<REPO>` placeholders, so it's
  unclear whether the push has been executed. **Verify both links open publicly.**
- **No benchmarked metrics exist in this repo.** All comparison figures must come
  from Repo 1 README / HF homepage; none are asserted here.
- **Module 2 (resume→fit, rules engine, gap analysis, DuckDB/LanceDB)** is Phase 2
  / in progress — described as planned, not shipped, in the points above.
