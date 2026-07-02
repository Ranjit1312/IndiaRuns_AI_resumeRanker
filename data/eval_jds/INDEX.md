# Eval JD dataset — INDEX

Gold JDs for the harness regression + multi-model parity eval. Each folder has
`jd.md`/`jd.txt` (verbatim), `jd_profile.yaml` (validated by
`redrob_ranker.profile.load`), `jd_meta.yaml` (coaching sidecar), `source.json`.

**10 validated golds:**

| slug | org | role | domain | validates |
|------|-----|------|--------|-----------|
| amazon_security-engineer-aws | Amazon | Security Engineer | cloud/app security | ✅ |
| amazon_senior-tpm-global-logistics | Amazon | Sr Technical Program Manager | TPM / logistics | ✅ |
| anthropic_product-manager-developer-productivity | Anthropic | Product Manager | product management | ✅ |
| anthropic_solutions-architect-applied-ai | Anthropic | Solutions Architect, Applied AI | pre-sales LLM | ✅ |
| databricks-solutions-architect | Databricks | Solutions Architect | pre-sales / data | ✅ |
| databricks_solutions-architect-digital-native | Databricks | SA, Digital Native | pre-sales / data | ✅ |
| databricks_resident-solutions-architect | Databricks | Sr Forward Deployed Engineer¹ | delivery eng | ✅ |
| google-data-scientist | Google | Business Data Scientist | analytics | ✅ |
| nvidia_solutions-architect-ai-ml | NVIDIA | Solutions Architect, AI/ML | AI/ML pre-sales | ✅ |
| stripe_backend-software-engineer | Stripe | Backend Engineer | payments/backend | ✅ |

¹ the requested "Resident Solutions Architect" Greenhouse id now resolves to a Sr. FDE posting; authored to the actual fetched JD.

## Status of the 10 user-requested links (2026-07-02 batch)

| # | role | outcome |
|---|------|---------|
| 1 | Google — Applied AI/ML Engineer, Finance DnA | ⛔ **JS-only SPA** — no static/WebFetch access |
| 3 | Google — SWE, ML Fleet Intelligence | ⛔ **JS-only SPA** |
| 6 | Anthropic — Forward Deployed Engineer, Applied AI | ⚠️ link closed; **captured current equivalent** "Solutions Architect, Applied AI" |
| 13 | Databricks — SA, Digital Native Business | ✅ captured (Greenhouse API) |
| 14 | Databricks — Resident Solutions Architect | ✅ captured (id now = Sr. FDE, CME&G) |
| 15 | Databricks — SA, Public Sector | ⛔ **HTTP 404** — posting closed |
| 19 | Microsoft — MTS, Applied Scientist (Copilot) | ⛔ **JS-only SPA** (microsoft.ai) |
| 21 | AWS — Solutions Architect, Associate | ⛔ **HTTP 404** — posting closed |
| 23 | AWS/Annapurna — ML Systems & Silicon | ⛔ **HTTP 404** — posting closed |
| 24 | NVIDIA — Solutions Architect, AI and ML | ✅ captured (Workday CXS API) |

**Recoverable with your help:** the 3 SPAs (Google ×2, Microsoft) need a JS
renderer — connect the Chrome extension (I'll render + read them) or paste the JD
text. The 3 HTTP-404 links are dead postings — send fresh links or approve
substitutes.

## Contract findings (validated across 10 real roles)
1. **`only_consulting` must be OFF for pre-sales / SA / FDE roles** — services &
   consulting experience is *desired*, not a red flag (Databricks ×3, NVIDIA,
   Anthropic SA).
2. **`stale_ic_role` is per-role:** ON only for hands-on IC coding (Databricks
   FDE, Amazon Security, Stripe); OFF for advisory/pre-sales/PM/TPM.
3. **`cv_primary` OFF for NVIDIA AI/ML** — computer-vision background is welcome
   there; the flag is only for roles that *reject* CV-primary careers.
4. Role-specific exclusions ride on **`domain.out_of_domain_terms`** (the general
   disqualifier lever); hard requirements the engine can't score (degree, travel,
   visa) live in `jd_meta.yaml` as `enforced_by: none` (advisory).

## Method notes (`tools/fetch_jd.py`)
- Greenhouse API (Anthropic/Databricks), Workday CXS (NVIDIA), Amazon `.json`,
  JSON-LD — never a summarizer, so JD text is verbatim. Greenhouse returns
  HTML-escaped content (unescape → strip tags).
- JS-only SPAs (Google careers, microsoft.ai) return 0 words of static HTML → need
  the Chrome MCP or a manual paste.
- Validate: `./.venv/Scripts/python.exe -m redrob_ranker.profile --check data/eval_jds/<slug>/jd_profile.yaml --method jd/method_config.yaml`
