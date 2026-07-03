# Phase 2 — implementation spec (resume→fit + gap analysis + DB layer)

This is the authoritative build spec for Module 2. It is grounded in the **actual
RedRob engine** (Repo 1), located at:
`C:\Users\jadha\Downloads\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\`
(referred to below as `REPO1/`). Read the cited files there before porting.

## Ground truth from Repo 1 (already inspected)
- **Candidate schema:** `REPO1/candidate_schema.json` — a candidate has
  `candidate_id`, `profile` (headline/summary/location/country/years_of_experience/
  current_title/company/size/industry), `career_history[]`
  (company/title/start_date/end_date/duration_months/is_current/industry/company_size/
  description), `education[]`, `skills[]` (name/proficiency/endorsements/duration_months),
  optional `certifications[]`/`languages[]`, and **23 `redrob_signals`** (platform
  activity: completeness, dates, response rates, `skill_assessment_scores`,
  github_activity_score, notice_period_days, expected_salary, work-mode, verified flags…).
- **`REPO1/redrob_ranker/intrinsic.py`** — `extract_intrinsic(records)->DataFrame`. PURE,
  per-candidate, no JD knowledge, numpy/pandas only. **Reuse verbatim.** It turns one
  candidate dict into the structured fact row (yoe, career_months, avg_tenure_months,
  n_jobs, max_role_months, skills_json, n_expert(_zero_dur), salary_min/max, anach,
  concurrent_deg, min_edu_end_year, assessments_json, last_active_date, signup_date,
  recruiter_response_rate, open_to_work_flag, interview_completion_rate,
  profile_completeness_score, notice_period_days, github_activity_score,
  willing_to_relocate, remote_pref, + 8 numeric signals + 3 bool flags).
- **`REPO1/redrob_ranker/features.py`** — the pool/artifact feature builder. We DO NOT
  reuse it wholesale (it needs precomputed pool embeddings + BM25 parquet + cross-candidate
  `reduceat`). We port its **per-candidate** formulas (line refs below).
- **`REPO1/redrob_ranker/rules.py`** — `compute_rules`: the composite + gated product.
  We port the composite math but DROP `mm()` (min-max needs a pool; meaningless for one
  candidate).
- **`REPO1/redrob_ranker/gates.py`** — the GATES column set (interpretability boundary).
- The AI System tree already has `redrob_ranker/profile.py` (the authoritative loader:
  `load(jd_path, method_path) -> (Profile, Method)`), `jd/method_config.yaml`,
  `jd/jd_profile.schema.json`. `Profile` exposes `.signals` (each `.id`, `.label`,
  `.query`, `.dense_weight`, `.evidence_re`), `.dense_extras`, `.domain` (in/out regex +
  terms), `.relevant_skill_re`, `.role` (peak_years/sigma_years/…), `.locations`,
  `.red_flag_enabled(name)`, `.signal_ids()`, `.evidence_signals()`. `Method` exposes all
  numeric knobs (`additive_weights`, `evidence_gate`, `claim_consistency`,
  `assessment_bonus`, `recency_ladder`, `experience_band`, `damps`, `recency`,
  `thresholds`, `evidence_context`, `integrity`, `availability`, `notice_tiers`,
  `location_ladder`, `hopper_def`, `lexicons`, `assessment`, `context_re`, `years_re`,
  `ref_date`, `hop_rate_tenure_months`).

## Design decision: candidate-centric single-candidate scorer (NOT pool ranking)
The engine ranks a POOL (`final_rules = mm(fit) * gates`). Our tool scores ONE resume for
coaching, so we output an **interpretable scorecard + gap analysis**, faithful to the
engine's per-candidate formulas, with NO pool normalization and NO precomputed artifacts.
Embeddings come live from `harness.backends.GoogleGenAIBackend.embed()` (gemini-embedding,
same BYO key). Lexical (BM25) is pool-defined, so we use a documented **single-candidate
lexical proxy** (keyword/token overlap of the signal query against the candidate's own
chunks) and clearly label it as an approximation of the pool BM25 channel.

---

## PART B — `store/` package (embedded DB layer, session-first, swappable)
Physically separate the three concerns (user requirement — "DB creation, schema creation,
DB updation separate, swappable to enterprise without fragmentation"):

- `store/schema.py` — **schema definitions only** (dataclasses + DDL strings) for tables:
  `workspaces, profiles, resumes, candidate_records, fit_runs, corrections, embeddings`.
  Pure data; NO connections.
- `store/base.py` — **the swap seam**: `RelationalStore` + `VectorStore` Protocols and a
  `Workspace` facade unifying them (methods: save/get/list profiles & resumes, save
  candidate_record, save fit_run, save correction-delta, upsert/search embeddings).
- `store/bootstrap.py` — **DB creation / migration only**: idempotent `create_duckdb(path)`,
  `create_lancedb(path)`, `ensure_schema(conn)`. Nothing else creates tables.
- `store/duckdb_store.py` — relational CRUD impl (lazy `import duckdb`).
- `store/lance_store.py` — vector upsert/search impl (lazy `import lancedb`); every vector
  row carries `embedding_model` + `dim`; one table per (model, dim) so EmbeddingGemma and
  Gemini vectors NEVER mix.
- `store/memory_store.py` — in-session impl backed by plain dicts (the **default**; HF-safe,
  no files, no deps). Implements the same `Workspace` facade.
- `store/__init__.py` — `get_store(config=None) -> Workspace` factory: returns `MemoryStore`
  by default; `STORE_BACKEND=duckdb+lance` (env/config) flips to the file-backed impls;
  enterprise back-ends later implement the same Protocols. Callers only touch the facade.
- **No new hard deps.** `duckdb`/`lancedb` are lazy-imported inside their modules; document
  them in `requirements-db.txt` (NOT the core `requirements.txt`, so the HF Space stays
  light). Add unit tests for `MemoryStore` round-trips (`tests/test_store.py`).

---

## PART C1 — engine reintroduction + single-candidate fit scorer
Files (in AI System `redrob_ranker/` and a new `redrob_ranker/fit.py`):
- **Copy `REPO1/redrob_ranker/intrinsic.py` → `redrob_ranker/intrinsic.py` verbatim.**
- **Copy `REPO1/candidate_schema.json` → `candidate_schema.json`** (repo root).
- **`redrob_ranker/fit.py`** — `score_candidate(candidate: dict, profile, method, backend,
  *, ref_date=None) -> FitResult`. Steps (port the cited per-candidate math faithfully;
  read every referenced line in REPO1):
  1. `intr = extract_intrinsic([candidate]).iloc[0]`.
  2. Build text chunks: each `career_history[i]` → `"{title} {company} {description}"`; plus
     `headline+summary`; skills names. (features.py builds `chunks`, `jobs_txt`, `head`.)
  3. **Dense per-signal** (features.py L217-236): `backend.embed` each `profile.signals[].query`
     (→ jd_vecs) and each job chunk (→ job_matrix) and the summary (→ summ_matrix), L2-normalize,
     cosine. Recency pooling: `msince` from job end_date to ref_date, `rw=0.5**(msince/facet_hl)`,
     `W=rw*sqrt(dur)`, `recencywt = Σ(sim*W)/ΣW` per signal. (facet_hl=`method.recency`.)
  4. **Evidence** (features.py L238-264): compile `method.context_re[internal|owner|scale]`,
     run per chunk, modifiers `ctx*own*sca*rec_e` (`method.evidence_context`), per evidence
     signal `evid = max over chunks(flag*mods)`; `evid_coverage = mean over evidence signals`;
     `depth_bonus` per L256-259.
  5. **Structured** (features.py L266-305): `cv_primary` (L279), `domain_nlp_ratio` (L286),
     `ai_skill_corroboration`+`ai_skills_claimed` (L292-300).
  6. **yoe_fit** gaussian (L323), **hopper** (L327 `method.hopper_def`), **only_consulting**
     +`months_since_ic_role` (L183-215), **loc2** location ladder (L330-363),
     **integrity** ladder (L365-394), **availability** (L397-415), **notice_pen** (L418-423),
     **assess_strength** (L308-318).
  7. **Composite** (rules.py L119-178) WITHOUT `mm()`: additive `fit = aw.dense*dense_fit +
     aw.lexical*lex_fit + aw.depth*depth_bonus` (lex_fit = single-candidate proxy, see above),
     then ×evidence_gate ×claim ×assess ×recency_ladder(if stale_ic enabled) ×experience_band
     ×red-flag damps(if enabled) ×location damp. Keep `integrity, availability, notice_pen`
     as separate reported multipliers (the gated product, minus mm).
  8. Return a `FitResult` dataclass: `overall` (0–100 interpretable fit index — define it
     explicitly as the product of the normalized component scores × gate multipliers; document
     the formula, do NOT call it a percentile), `per_signal` (id, label, dense, lexical,
     evidence, weight, contribution), `gates` (integrity/availability/notice/location with
     the value and whether it damped), `red_flags` (which enabled flags fired: cv_primary/
     hopper/only_consulting/stale_ic), and `gaps` (human-readable coaching: each low signal
     and each damping gate → what to change; map signal_id→jd_meta must_have/nice_to_have
     text where available). Provide `to_dict()`.
- **Requirements:** add `numpy`, `pandas` to core `requirements.txt` (needed by
  intrinsic/fit; both are light, no torch). Keep everything else as-is.
- **Tests** (`tests/test_fit.py`): with `MockBackend` returning deterministic vectors, score
  a hand-built candidate against a gold profile; assert `FitResult` structure, that a strong
  candidate scores > a weak one, and that an enabled red flag / short-notice gate shows up in
  `gaps`. Must not require a live API.

## PART C2 — resume→candidate harness (`harness/resume.py`)
Mirror `harness/coerce.py`'s RLM discipline:
- `compile_resume(resume_text, backend, *, logger=None, on_event=None) -> ResumeResult`
  (candidate dict + health + validation + telemetry, reusing `harness.rlm.llm_query`,
  `harness.logging_utils`). Leaf extractions → `profile` block, `career_history[]`,
  `education[]`, `skills[]`. **Projects → `profile.summary`, NOT `career_history`** (avoids
  tenure/hopper distortion — project the memory rule).
- `harness/candidate_fields.py` — sentinels + sanitizers for the 23 `redrob_signals` a resume
  can't supply, and a `validate_candidate(dict)` against `candidate_schema.json` (jsonschema),
  mirroring `harness/validate.py`. Sentinel policy (so gates stay NEUTRAL, not punitive):
  `last_active_date` = today's ref_date (avoids dormancy/availability damp),
  `signup_date` = a year before, `recruiter_response_rate` = -1 (missing→neutral),
  `interview_completion_rate` = 0.5, `profile_completeness_score` = 50,
  `skill_assessment_scores` = {} (assess_strength=0 → m_assess=1.0, neutral),
  `github_activity_score` = -1 unless a GitHub is parsed, `open_to_work_flag` = true,
  `notice_period_days` from resume or 60, verified flags false, salary {min:0,max:0}.
  Generate a valid `candidate_id` (`CAND_` + 7 digits).
- Reuse the RLM leaf/telemetry pattern so the resume compile ALSO streams live (Part A).

## PART C3 — synthetic resume generator (`tools/gen_resumes.py`)
- ~12 resumes aligned to the 10 gold JDs in `data/eval_jds/`, varied fit
  (strong/borderline/weak archetypes), **placeholder PII only** (`"XYZ Candidate"`,
  `abc@example.com`, `+91-XXXXXXXXXX`), deterministic `random.seed`. Seed each resume off its
  gold JD's `jd_meta.yaml` must_haves/signals so fit tiers are realistic.
- Export **txt + DOCX** (`python-docx`, already a dep) and **PDF** (`fpdf2` — add to
  `requirements-db.txt`/a dev extra, lazy-import; skip PDF gracefully if absent) into
  `data/synthetic_resumes/<jd_slug>_<tier>/resume.(txt|docx|pdf)`. Write an `INDEX.md`.

## PART C4 — HITL app wiring (`app.py` → multipage, + fit UI)
- Add a **Resume → Fit** page/tab: upload PDF/DOCX/txt (reuse `extract_text`) → `compile_resume`
  (live telemetry) → **prefilled editable HITL form** (profile fields; career_history rows;
  education; skills) + advisory weight sliders for missing signals → user approves/corrects/adds.
  Capture the parsed-vs-approved **delta** and persist via `store` (`corrections`).
- On approve: `validate_candidate` → `score_candidate` against the compiled/loaded JD profile →
  render the **scorecard** (overall fit index, per-signal bars, gates, red-flags) and the
  **gap analysis / coaching** list. Store the `fit_run`.
- **Multi-profile workspace:** sidebar lists workspace profiles+resumes from `store` (session
  `MemoryStore` by default). Keep BYO-key/session-only/no-key-logging contract.

---

## Sequencing & ownership (build via Sonnet subagents)
1. **Agent A** — Part B `store/` (independent).
2. **Agent B** — Part C1 engine+fit (`redrob_ranker/intrinsic.py`, `candidate_schema.json`,
   `redrob_ranker/fit.py`, `tests/test_fit.py`, requirements). Independent of A.
3. **Agent C** — Part C2 resume harness + Part C3 synthetic generator (needs B's schema).
4. **Agent D** — Part C4 HITL app + store wiring (needs A, B, C).
Each agent: read the cited REPO1 files, match existing code style, add tests, run
`./.venv/Scripts/python -m pytest tests/ -q` green, AST-parse changed files, NO git commit,
do not run streamlit.

## Faithfulness rules
- Every numeric constant comes from `profile`/`method` (via `redrob_ranker.profile.load`) —
  nothing hardcoded that the engine reads from config.
- Red-flag damps apply ONLY when `profile.red_flag_enabled(name)` (per-JD toggle).
- Do not reuse Repo 1's precomputed pool artifacts or `mm()`; single-candidate only.
- Embeddings: one model per run (gemini-embedding via `backend.embed`); never mix vector
  models in one LanceDB table.
