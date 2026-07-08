# CONTEXT — domain glossary

Domain vocabulary for the Resume-Fit tool. Architecture reviews and design
conversations use these nouns so modules are named after concepts, not mechanics.
(Architecture vocabulary — module / interface / seam / adapter / depth — lives in the
`improve-codebase-architecture` skill's LANGUAGE.md, not here.)

## Core artifacts
- **JD profile** — the schema-valid `jd_profile.yaml` that configures the RedRob ranking
  engine for one job. Produced by the **JD compiler** (`harness/coerce.py`).
- **JD meta** — the coaching sidecar `jd_meta.yaml` (must-haves, nice-to-haves, advisory
  exclusions) that rides alongside a JD profile.
- **Candidate** — the schema-valid candidate dict (`candidate_schema.json`): profile,
  career history, education, skills, and the 23 `redrob_signals`.
- **Parsed résumé** — a **candidate** as first extracted from résumé text, *before* human
  correction. The output of `parse_resume`. Platform-only signals are neutral sentinels.

## Scoring
- **Fit index** — the interpretable 0–100 score for one candidate against one JD profile.
  NOT a percentile; the normalized dense/lexical/evidence composite × gate multipliers.
- **Signal** — one scored criterion of a JD profile (dense weight + optional evidence
  regex). A candidate scores per-signal.
- **Gate** — a multiplier applied outside the composite: integrity, availability, notice,
  location. A **damped** gate pulls the fit index down.
- **Red flag** — one of four per-JD toggles (`cv_primary`, `job_hopper`,
  `only_consulting`, `stale_ic_role`) that fire only when the JD enables them.
- **Gap analysis** — the human-readable coaching derived from low signals + damped gates.
- **Fit run** — one scoring of an (approved candidate, JD profile) pair. Persisted as a
  `fit_run` row in the store; the unit the Dashboard counts.

## Flow / orchestration (Candidate B — the deepened seam)
- **Fit session** — the lifecycle of turning one résumé into a **fit run**:
  parse → (human correction) → validate → score → persist. Lives in `fit_session.py`,
  above the RLM harness, the fit scorer, and the store.
  - `parse_resume(text, backend, *, on_event) → ParsedResume` — everything before the HITL
    form. Deep: owns extract-or-ocr + `compile_resume` + telemetry.
  - `run_fit(candidate, jd_profile_yaml, backend, store, *, on_event) → FitOutcome` —
    everything after approval: validate → load Profile/Method → `score_candidate` →
    persist the fit run. Owns persistence (store injected).
- **Fit outcome** — the typed result of `run_fit`: a `status` ∈ {ok, rate_limited,
  transient, invalid}, plus the fit result, persisted ids, and telemetry. The UI renders
  uniformly off `status` instead of catching exceptions.

## Machinery (not domain, but load-bearing)
- **RLM harness** — the depth-1 Recursive-Language-Model compile loop: leaf extractions →
  assemble → validate → repair-the-failing-block → sentinel-degrade.
- **Compile loop** (Candidate A) — `rlm.run_compile(env, spec, backend, *, logger,
  on_event, max_repairs) → CompileResult`: the one deep module that owns the
  "never emit an invalid artifact" invariant + telemetry/health. Both compilers become
  thin specs.
- **Artifact spec** — what `run_compile` needs to compile one artifact type: an ordered
  builder set, an `assemble(parts, env)→dict` hook (where JD's `cross_encoder_query` /
  résumé's projects→summary / sentinel-signals live), a **validator**, a
  `rebuild(dict, failing_top, …)→dict` (owns special cases, e.g. JD refreshing
  `cross_encoder_query` when `role` changes), a sentinel map, and an optional
  `finalize` (JD's `jd_meta`). Two adapters: `JDSpec`, `ResumeSpec`.
- **Validator** (Candidate C) — the seam `(dict) → ValidationResult`. Two adapters:
  `EngineProfileValidator` (temp-file + `redrob_ranker.profile.load`) and
  `JsonSchemaValidator` (`jsonschema` Draft-7 against `candidate_schema.json`). The
  artifact spec injects one; `run_compile` never knows which mechanism runs.
- **Ingest seam** — `harness/ingest.py`: the one place a file (PDF/DOCX/text/Google Doc)
  becomes text, incl. the Gemma-multimodal OCR fallback.
- **Store / Workspace** — the swappable persistence facade (`MemoryStore` default;
  DuckDB/LanceDB adapters). The exemplar seam the rest of the codebase emulates.
- **Backend seam** (Candidate E) — the model provider, split into two capability
  interfaces: `TextBackend` (`generate` + `generate_multimodal` + `last_usage`) and
  `EmbeddingBackend` (`embed`). `GoogleGenAIBackend` and `MockBackend` both satisfy both,
  so mock and real can't drift; callers depend on the narrower interface they use (the
  fit scorer needs `EmbeddingBackend`; the RLM leaves need `TextBackend`).
- **Fit parity fixture** (Candidate D) — a frozen golden of the fit scorer's deterministic
  per-candidate features + gates (everything except the single-candidate dense/lexical
  approximations), derived from Repo 1's `features.py`/`rules.py`. `test_fit_parity.py`
  asserts `fit.py` still reproduces it, so the hand-port can't silently drift from its
  source of truth; a regen script (guarded on the sibling Repo 1 being present) refreshes
  it deliberately.
