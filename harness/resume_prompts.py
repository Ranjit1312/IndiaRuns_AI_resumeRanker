"""RLM leaf prompts for the resume->candidate seam (Phase 2, Part C2).

Same discipline as `harness/prompts.py`: each leaf asks for ONE field group
from a focused resume slice, JSON-only. The harness parses tolerantly
(`harness.jsonutil.extract_json`) and validates authoritatively against
`candidate_schema.json`, so prose leakage is survivable.

Faithfulness rule (docs/PHASE2_SPEC.md): projects must NOT be extracted into
career_history (they would distort tenure/hopper math in redrob_ranker.fit).
The `projects_prompt` below is deliberately a separate leaf whose output the
root (`harness/resume.py`) appends only to `profile.summary`.
"""
from __future__ import annotations

SYSTEM = (
    "You extract structured career facts from a resume/CV. "
    "You output ONLY the requested JSON — no prose, no code fences, no comments. "
    "Extract labelled variables; do not reason or editorialize. Do not invent "
    "facts not present or reasonably implied by the resume text."
)


def profile_prompt(snippet: str) -> str:
    return f"""From the resume below, extract the candidate's profile summary as JSON:
{{"headline": str (one-line professional headline, e.g. current title + focus),
  "summary": str (2-4 sentence professional summary, resume's own words where possible),
  "location": str (city, region/state — "Unknown" if absent),
  "country": str (default "India" if not stated),
  "years_of_experience": number (total professional years; estimate from earliest
    to latest role dates if not explicitly stated),
  "current_title": str, "current_company": str,
  "current_company_size": one of ["1-10","11-50","51-200","201-500","501-1000",
    "1001-5000","5001-10000","10001+"] (best estimate; "201-500" if unknown),
  "current_industry": str}}

RESUME:
{snippet}

JSON:"""


def career_history_prompt(snippet: str) -> str:
    return f"""From the resume below, extract EVERY employment/work-experience entry
(NOT internships-only unless that's all there is, NOT projects, NOT education) as
a JSON array, most recent first:
[{{"company": str, "title": str,
   "start_date": "YYYY-MM-DD" (use "01" for unknown day/month if only year/month given),
   "end_date": "YYYY-MM-DD" or null (null means current/ongoing role),
   "duration_months": integer (compute from dates if not stated),
   "is_current": bool,
   "industry": str (best guess from company/role context),
   "company_size": one of ["1-10","11-50","51-200","201-500","501-1000",
     "1001-5000","5001-10000","10001+"] (best estimate),
   "description": str (role responsibilities/achievements, 1-3 sentences,
     summarizing the resume's own bullets)}}]
Only include real employment roles. Do NOT include standalone/personal/academic
projects here — those are handled separately.

RESUME:
{snippet}

JSON array:"""


def education_prompt(snippet: str) -> str:
    return f"""From the resume below, extract every education entry as a JSON array:
[{{"institution": str, "degree": str, "field_of_study": str,
   "start_year": integer, "end_year": integer,
   "grade": str or null (GPA/percentage/class if stated)}}]
Use [] if no education is mentioned.

RESUME:
{snippet}

JSON array:"""


def skills_prompt(snippet: str) -> str:
    return f"""From the resume below, extract the candidate's skills as a JSON array:
[{{"name": str (skill/tool/technology name),
   "proficiency": one of ["beginner","intermediate","advanced","expert"]
     (infer from years used / seniority / how it's described; default "intermediate"),
   "endorsements": integer (0 unless the resume states endorsement/rating counts),
   "duration_months": integer or null (how long they've used it, if inferable)}}]
Deduplicate; keep the resume's most specific/technical skill names.

RESUME:
{snippet}

JSON array:"""


def projects_prompt(snippet: str) -> str:
    """Separate leaf for personal/academic/side projects.

    Output feeds ONLY `profile.summary` (never career_history) — see module
    docstring's faithfulness rule.
    """
    return f"""From the resume below, extract a short paragraph (2-5 sentences)
summarizing the candidate's standalone PROJECTS (personal, academic, open-source,
side/hackathon projects — NOT employer work experience). Output JSON:
{{"projects_summary": str}}
If there are no projects section / no standalone projects, return {{"projects_summary": ""}}.
Do not include employment history here.

RESUME:
{snippet}

JSON:"""
