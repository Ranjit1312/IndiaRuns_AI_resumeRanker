# Eval JD dataset — INDEX

6 cross-domain real JDs, each with a gold `jd_profile.yaml` (validated by
`redrob_ranker.profile.load`) + coaching `jd_meta.yaml`. Seed set for the harness
regression + multi-model parity eval. (Collection capped at 10 for now; the
earlier fan-out's empty folders were pruned.)

| slug | org | role | domain | #signals | validates |
|------|-----|------|--------|----------|-----------|
| amazon_security-engineer-aws | Amazon (AWS) | Security Engineer | cloud/app security | 6 | ✅ OK |
| amazon_senior-tpm-global-logistics | Amazon | Sr Technical Program Manager | TPM / logistics | 5 | ✅ OK |
| anthropic_product-manager-developer-productivity | Anthropic | Product Manager | product management | 6 | ✅ OK |
| databricks-solutions-architect | Databricks | Solutions Architect | pre-sales / data platform | 6 | ✅ OK |
| google-data-scientist | Google | Business Data Scientist | analytics / experimentation | 6 | ✅ OK |
| stripe_backend-software-engineer | Stripe | Backend Engineer | payments / backend | 6 | ✅ OK |

Roles span: security eng, program mgmt, product mgmt, pre-sales architecture,
data science, backend eng — a genuine cross-domain spread.

## Contract findings (validated on real roles)
1. **`stale_ic_role` / `cv_primary` are role-specific, not universal.** Off for
   TPM and PM (penalizing "no recent IC coding" is wrong for non-IC roles); on for
   engineering roles. `cv_primary` irrelevant to most non-AI roles. → toggle the 4
   gates per-JD; use `domain.out_of_domain_terms` as the general disqualifier lever.
2. **Non-engineering roles have softer `evidence_regex` fingerprints** (PM, data
   scientist "exec storytelling") — signals still work via the dense `query`; set
   `evidence_regex: null` where there's no clean keyword fingerprint.
3. **Hard requirements the engine can't score** (degrees, KPI ownership, work
   auth) → `jd_meta.yaml` with `enforced_by: none` (advisory). Exactly the
   enforced-vs-advisory split the coaching layer needs.
4. **SPA career sites** (metacareers.com, microsoft.com) are client-hydrated and
   unfetchable statically → substitute a comparable real posting (e.g. Anthropic PM
   for Meta PM) or `curl` server-rendered sites (amazon.jobs, stripe, databricks).

## Method notes
- Validate: `./.venv/Scripts/python.exe -m redrob_ranker.profile --check data/eval_jds/<slug>/jd_profile.yaml --method jd/method_config.yaml`
- `_progress/*.md` are the live collection logs.
