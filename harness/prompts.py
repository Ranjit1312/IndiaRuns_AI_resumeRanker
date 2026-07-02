"""RLM leaf prompts — each asks the model for ONE field group from ONE snippet.

Small models are far more reliable extracting a single labelled structure from a
focused slice than one-shotting the whole schema. Every prompt demands JSON only;
the harness parses tolerantly and validates authoritatively, so prose leakage is
survivable. Guidance mirrors jd/RETARGETING.md.
"""
from __future__ import annotations

SYSTEM = (
    "You extract structured hiring criteria from a job description. "
    "You output ONLY the requested JSON — no prose, no code fences, no comments. "
    "Extract labelled variables; do not reason or editorialize."
)


def role_prompt(snippet: str) -> str:
    return f"""From the job description below, extract the role block as JSON:
{{"title": str, "company": str, "domain": str (short label, e.g. "data engineering"),
  "summary": str (1-2 sentences), "min_years": number, "max_years": number,
  "peak_years": number, "sigma_years": number, "notice_preference_days": integer}}
Rules: years from any "X-Y years" phrasing; peak_years = the ideal (midpoint or stated);
sigma_years ~ (max-min)/2 (>=1). If notice period is not stated, use 60. If YoE is not
stated, estimate from seniority (junior 1-3, mid 3-6, senior 5-9, staff 8-14).

JOB DESCRIPTION:
{snippet}

JSON:"""


def locations_prompt(snippet: str) -> str:
    return f"""From the job description below, extract locations as JSON:
{{"preferred": [lowercase city tokens], "acceptable": [lowercase city tokens],
  "relocation_acceptable": bool, "remote_acceptable": bool}}
Use [] for unknown city lists. remote_acceptable true only if remote/hybrid is offered.

JOB DESCRIPTION:
{snippet}

JSON:"""


def signal_labels_prompt(snippet: str) -> str:
    return f"""List the 4-7 MUST-HAVE capability axes for this role — the "you
absolutely need this" skills/experiences. Output ONLY a JSON array of short
lowercase phrases (2-4 words each), most important first.

JOB DESCRIPTION:
{snippet}

JSON array:"""


def signal_detail_prompt(label: str, snippet: str) -> str:
    return f"""For the capability "{label}" required by the role below, output JSON:
{{"query": str (a short positive phrase describing what good looks like — what to
   search a candidate's history for),
 "evidence_regex": str OR null (a case-insensitive Python regex of concrete
   terms/tools that leave a fingerprint in career text, e.g.
   "kubernetes|k8s|helm|terraform"; use null if this is a soft axis with no clean
   keyword fingerprint),
 "dense_weight": number 0.05-0.30 (how central this capability is)}}

JOB DESCRIPTION:
{snippet}

JSON:"""


def domain_prompt(snippet: str) -> str:
    return f"""Define what makes a candidate IN-domain vs OUT-of-domain for this
role. Output JSON:
{{"in_domain_terms": [lowercase terms that signal a good-fit background],
  "out_of_domain_terms": [lowercase terms for backgrounds this role does NOT want
   — this is how we disqualify wrong fits],
  "in_domain_regex": str (case-insensitive regex over the in-domain terms),
  "out_of_domain_regex": str (case-insensitive regex over the out-of-domain terms)}}
Provide at least 3 terms per list and valid regexes.

JOB DESCRIPTION:
{snippet}

JSON:"""


def relevant_skill_prompt(snippet: str) -> str:
    return f"""Output JSON {{"relevant_skill_regex": str}} — a single case-insensitive
Python regex matching the skill names that count for THIS role (tools, languages,
methods, and relevant soft skills), alternated with "|".

JOB DESCRIPTION:
{snippet}

JSON:"""


def red_flags_prompt(snippet: str) -> str:
    return f"""Decide which of these 4 supported red-flag gates THIS role cares about.
Output JSON {{"cv_primary": bool, "job_hopper": bool, "only_consulting": bool,
"stale_ic_role": bool}} where:
- cv_primary: true only if the role rejects computer-vision/speech-primary backgrounds
- job_hopper: true if stable tenure matters (most engineering roles)
- only_consulting: true if the role wants product (not pure services/consulting) experience
- stale_ic_role: true ONLY for hands-on individual-contributor coding roles; FALSE for
  managers, program/product managers, architects-advisory, writers, designers.

JOB DESCRIPTION:
{snippet}

JSON:"""


def meta_extras_prompt(snippet: str) -> str:
    """One leaf for the coaching sidecar's non-signal parts."""
    return f"""From the job description, extract JSON for a candidate-coaching sidecar:
{{"hard_requirements": [{{"text": str, "kind": "yoe"|"education"|"auth"|"location"|
   "certification"|"skill"}}],
 "explicit_exclusions": [{{"text": str}}]  // things the JD explicitly does NOT want,
 "coaching_notes": str (1-2 actionable sentences for an applicant)}}
Use [] when none are stated.

JOB DESCRIPTION:
{snippet}

JSON:"""
