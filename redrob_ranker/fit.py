"""fit.py — single-candidate interpretable fit scorer (Phase 2, Part C1).

`score_candidate(candidate, profile, method, backend, *, ref_date=None) ->
FitResult` is a candidate-centric sibling of Repo 1's pool ranker
(`redrob_ranker/features.py` + `redrob_ranker/rules.py`). It scores ONE resume
against ONE JD for coaching, not a pool for ranking, so it deliberately:

  - does NOT call `rules.mm()` — min-max needs a pool; meaningless for one row.
  - does NOT read Repo 1's precomputed artifacts (job_embeddings.npy,
    bm25_facets.parquet, jd_vectors.npy, evidence_texts.parquet). Embeddings
    are produced live via `backend.embed()`; the JD lexical (BM25) channel —
    which is pool-defined (BM25 needs a corpus of jobs to compute IDF against)
    — is replaced by a documented single-candidate lexical proxy (see
    `_lexical_proxy` below): normalized token/keyword overlap between each
    signal's `query` and the candidate's own text chunks. This is clearly an
    approximation of the real BM25 channel, not a faithful port of it.

Every other formula below is a faithful, line-referenced port of the
per-candidate math in REPO1 `redrob_ranker/features.py` and
`redrob_ranker/rules.py` (line numbers refer to the REPO1 files as read for
this port):

  - intrinsic row:            intrinsic.extract_intrinsic([candidate]).iloc[0]
  - dense per-signal + pooling: features.py L217-236
  - evidence coverage + depth_bonus: features.py L238-264 (method.context_re,
    method.evidence_context)
  - cv_primary / domain_nlp_ratio / ai corroboration: features.py L266-305
  - yoe_fit gaussian: features.py L323
  - hopper: features.py L327 (method.hopper_def)
  - only_consulting + months_since_ic_role: features.py L183-215
  - loc2 (location ladder): features.py L330-363
  - integrity ladder: features.py L365-394
  - availability: features.py L397-415
  - notice_pen: features.py L418-423
  - assess_strength: features.py L308-318
  - composite (WITHOUT mm()): rules.py L119-178

All numeric constants come from `profile` / `method` (via
`redrob_ranker.profile.load`) — nothing here hardcodes a value the engine
reads from config. Red-flag damps apply only when
`profile.red_flag_enabled(name)` is True (per-JD toggle).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np

from .intrinsic import extract_intrinsic

SEP = "\x1f"   # mirrors features.py's per-job chunk separator


# ---------------------------------------------------------------------------
# date helpers — verbatim logic from REPO1 features.py L53-64
# ---------------------------------------------------------------------------
def _parse_date(d):
    if not d:
        return None
    try:
        y, m, day = (int(x) for x in str(d)[:10].split("-"))
        return date(y, m, day)
    except Exception:
        return None


def _months_between(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.0


# ---------------------------------------------------------------------------
# result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalScore:
    id: str
    label: str
    dense: float          # cosine recency-pooled similarity, signal query vs career text
    lexical: float         # single-candidate lexical proxy (see module docstring)
    evidence: float        # evid_<id>: max over chunks(flag*ctx*owner*scale*recency), 0 if no evidence_regex
    weight: float           # profile signal.dense_weight
    contribution: float     # weight * dense — this signal's share of dense_fit


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    damped: bool            # True if this gate multiplier meaningfully reduced fit (< 0.999)


@dataclass(frozen=True)
class FitResult:
    """Interpretable single-candidate scorecard.

    `overall` (0-100) is an EXPLICITLY DEFINED interpretable fit index, NOT a
    percentile and NOT comparable across different JD profiles or candidate
    pools. It is defined as:

        overall = 100 * clip(dense_component, 0, 1)
                       * evidence_gate_mult
                       * claim_consistency_mult
                       * assessment_bonus_mult
                       * recency_ladder_mult (if stale_ic_role enabled)
                       * experience_band_mult
                       * red_flag_damp_product (only enabled flags)
                       * location_damp_mult
                       * integrity_mult
                       * availability_mult
                       * notice_mult

    where `dense_component` is the normalized additive composite
    (aw.dense*dense_fit + aw.lexical*lex_fit + aw.depth*depth_bonus) rescaled
    into [0, 1] by dividing by the maximum value the additive channels could
    reach (sum of all dense weights + yoe_fit_weight + domain_ratio_weight,
    weighted by aw.dense; plus aw.lexical + aw.depth, since lex_fit and
    depth_bonus are each already in [0, 1]). This keeps `overall` on an
    absolute, JD-and-mechanism-relative 0-100 scale (100 = every dense/lexical/
    depth channel maxed AND every multiplicative gate neutral) rather than a
    pool percentile, which the single-candidate scorer has no pool to compute.
    """
    overall: float
    per_signal: list          # list[SignalScore]
    gates: list                # list[GateResult] — integrity/availability/notice/location
    red_flags: list            # list[str] — enabled flags that fired
    gaps: list                 # list[str] — human-readable coaching strings
    candidate_id: str
    role_title: str

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "candidate_id": self.candidate_id,
            "role_title": self.role_title,
            "per_signal": [
                {"id": s.id, "label": s.label, "dense": s.dense,
                 "lexical": s.lexical, "evidence": s.evidence,
                 "weight": s.weight, "contribution": s.contribution}
                for s in self.per_signal
            ],
            "gates": [
                {"name": g.name, "value": g.value, "damped": g.damped}
                for g in self.gates
            ],
            "red_flags": list(self.red_flags),
            "gaps": list(self.gaps),
        }


# ---------------------------------------------------------------------------
# text chunk assembly — mirrors features.py's `chunks` / `jobs_txt` / `head`
# ---------------------------------------------------------------------------
def _build_chunks(candidate: dict) -> "tuple[list[str], list[dict], str]":
    """Return (job_chunks, job_metas, headline_summary_text).

    job_chunks[i] == "{title} {company} {description}" for career_history[i]
    (features.py builds `chunks` from `jobs_text`/SEP-joined per-job text built
    at JD-compile time; here we build the equivalent directly from the raw
    candidate dict since there is no precomputed evidence_texts.parquet).
    """
    jobs = candidate.get("career_history") or []
    chunks, metas = [], []
    for j in jobs:
        title = j.get("title") or ""
        company = j.get("company") or ""
        desc = j.get("description") or ""
        chunks.append(f"{title} {company} {desc}".strip())
        metas.append(j)
    p = candidate.get("profile") or {}
    head = f"{p.get('headline') or ''} {p.get('summary') or ''}".strip()
    return chunks, metas, head


# ---------------------------------------------------------------------------
# lexical proxy — SINGLE-CANDIDATE approximation of the pool BM25 channel.
#
# The real engine's lex_fit is the mean, over ALL signals, of a BM25 score
# computed against the whole candidate POOL's job-chunk corpus (IDF needs a
# corpus). We have exactly one candidate, so there is no corpus to compute
# IDF against. This proxy instead measures, per signal, the fraction of the
# signal query's distinctive tokens that appear anywhere in the candidate's
# own chunks (headline+summary+job text+skills) — a normalized token/keyword
# overlap in [0, 1]. It is clearly NOT BM25 and is documented as an
# approximation of the pool lexical channel per docs/PHASE2_SPEC.md.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "that", "this", "at", "by", "as", "be", "it", "from",
}


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS and len(t) > 1}


def _lexical_proxy(query: str, candidate_text: str) -> float:
    q_toks = _tokens(query)
    if not q_toks:
        return 0.0
    c_toks = _tokens(candidate_text)
    if not c_toks:
        return 0.0
    hits = sum(1 for t in q_toks if t in c_toks)
    return hits / len(q_toks)


# ---------------------------------------------------------------------------
# embedding helpers
# ---------------------------------------------------------------------------
def _require_embed(backend) -> None:
    if not hasattr(backend, "embed") or not callable(getattr(backend, "embed")):
        raise ValueError(
            "score_candidate needs live embeddings: backend "
            f"{getattr(backend, 'name', backend)!r} has no embed(texts) method. "
            "Use a backend that exposes embed() (e.g. GoogleGenAIBackend) or a "
            "test MockEmbedBackend.")


def _embed_matrix(backend, texts: list) -> "np.ndarray":
    """Embed `texts`, L2-normalize each row. Batches in ONE call to backend.embed."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)
    vecs = backend.embed(list(texts))
    mat = np.asarray(vecs, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# core scorer
# ---------------------------------------------------------------------------
def score_candidate(candidate: dict, profile, method, backend, *,
                     ref_date: "str | None" = None) -> FitResult:
    """Score ONE candidate against ONE JD (profile/method). See module docstring."""
    _require_embed(backend)

    REF = _parse_date(ref_date or method.ref_date)
    if REF is None:
        raise ValueError(f"fit: could not parse ref_date {ref_date or method.ref_date!r}")

    intr = extract_intrinsic([candidate]).iloc[0]
    cid = candidate.get("candidate_id", "")

    chunks, job_metas, head_text = _build_chunks(candidate)
    n_jobs = len(chunks)

    # -- recency / duration weights per job, mirrors features.py L176-214 -----
    msince = np.zeros(n_jobs)
    dur = np.ones(n_jobs)
    for i, j in enumerate(job_metas):
        d = j.get("duration_months") or 0
        dur[i] = max(1, d)
        if not j.get("is_current") and j.get("end_date"):
            ed = _parse_date(j["end_date"])
            if ed is not None:
                msince[i] = max(0.0, _months_between(ed, REF))

    facet_hl = method.recency["facet_halflife_months"]
    evid_hl = method.recency["evidence_halflife_months"]
    rw = 0.5 ** (msince / facet_hl)
    W = rw * np.sqrt(dur)
    wsum = float(W.sum()) if n_jobs else 0.0

    # -- embed: signal queries + job chunks + headline/summary (ONE batch) ---
    sig_ids = profile.signal_ids()
    n_sig = len(sig_ids)
    queries = [s.query for s in profile.signals]
    texts_to_embed = list(queries) + list(chunks) + [head_text]
    all_vecs = _embed_matrix(backend, texts_to_embed)
    jd_vecs = all_vecs[:n_sig]                                  # (NSIG, d)
    job_matrix = all_vecs[n_sig:n_sig + n_jobs]                 # (n_jobs, d)
    summ_vec = all_vecs[n_sig + n_jobs:n_sig + n_jobs + 1]      # (1, d)

    if n_jobs:
        simsJ = job_matrix @ jd_vecs.T                          # (n_jobs, NSIG)
    else:
        simsJ = np.zeros((0, n_sig))
    simsS = (summ_vec @ jd_vecs.T)[0] if n_sig else np.zeros(0)  # (NSIG,)

    if n_jobs and wsum > 0:
        recencywt = (simsJ * W[:, None]).sum(axis=0) / wsum      # (NSIG,)
    else:
        recencywt = np.zeros(n_sig)

    # -- evidence regexes per job chunk (features.py L238-259) ---------------
    INTERNAL_RE = method.context_re["internal"]
    OWNER_RE = method.context_re["owner"]
    SCALE_RE = method.context_re["scale"]
    EC = method.evidence_context

    def _flags(rx):
        return np.fromiter((1.0 if rx.search(c) else 0.0 for c in chunks),
                           np.float64, n_jobs)

    f_int = _flags(INTERNAL_RE); f_own = _flags(OWNER_RE); f_sca = _flags(SCALE_RE)
    ctx = np.where(f_int > 0, EC["internal"], 1.0)
    own = np.where(f_own > 0, 1.0, EC["no_owner"])
    sca = np.where(f_sca > 0, 1.0, EC["no_scale"])
    rec_e = 0.5 ** (msince / evid_hl)
    mods = ctx * own * sca * rec_e

    evid_signals = profile.evidence_signals()
    evid_by_id: dict = {}
    for s in profile.signals:
        if s.evidence_re is None:
            evid_by_id[s.id] = 0.0
            continue
        flat = _flags(s.evidence_re) * mods
        evid_by_id[s.id] = float(flat.max()) if n_jobs else 0.0
    evid_coverage = (float(np.mean([evid_by_id[s.id] for s in evid_signals]))
                     if evid_signals else 0.0)

    depth_flat = ((f_int == 0) & (f_own > 0) & (f_sca > 0)) * rec_e
    depth_bonus = float(depth_flat.max()) if n_jobs else 0.0

    # -- structured text features (features.py L266-305) ---------------------
    jt_all = " ".join(chunks)
    CV_RE = profile.domain.out_of_domain_re
    NLP_RE = profile.domain.in_domain_re
    NLPT = profile.domain.in_domain_terms
    CVT = profile.domain.out_of_domain_terms
    cv_n = len(CV_RE.findall(jt_all)); nlp_n = len(NLP_RE.findall(jt_all))
    cv_primary = float(cv_n >= 3 and cv_n > nlp_n)

    skills = candidate.get("skills") or []
    names = [(s.get("name") or "").lower() for s in skills]
    blob = (head_text + " " + jt_all + " " + " ".join(names)).lower()
    nlp_c = sum(blob.count(tm) for tm in NLPT)
    cv_c = sum(blob.count(tm) for tm in CVT)
    domain_nlp_ratio = (nlp_c + 1) / (nlp_c + cv_c + 2)

    YEARS_RE = method.years_re
    m = YEARS_RE.search(head_text or "")
    summary_years = float(m.group(1)) if m else None

    AITERMS = method.lexicons["ai_skill_terms"]
    evidence_l = (head_text + " " + jt_all).lower()
    claimed = [nm for nm in names if any(tm in nm for tm in AITERMS)]
    supported = 0
    for nm in claimed:
        toks = [w for w in re.split(r"[^a-z]+", nm) if len(w) >= 4]
        if any(w[:5] in evidence_l for w in toks):
            supported += 1
    ai_corr = supported / (len(claimed) + 1.0)
    ai_claimed_n = len(claimed)

    SD = method.integrity["skill_dur"]
    career_m = float(intr["career_months"])
    n_skill_dur_exceed = sum(
        1 for s in skills
        if (s.get("duration_months") or 0) > career_m * SD["ratio"] + SD["slack_months"])

    # -- assessments (features.py L308-318) -----------------------------------
    A = method.assessment
    DESIRED_RE = profile.relevant_skill_re
    sas = (candidate.get("redrob_signals") or {}).get("skill_assessment_scores") or {}
    rel = [v for k, v in sas.items() if DESIRED_RE.search(k)]
    if rel:
        topk = sorted(rel, reverse=True)[:A["top_k"]]
        st = min(1.0, max(0.0, (float(np.mean(topk)) - A["score_floor"]) / A["score_span"]))
        assess_strength = st * len(rel) / (len(rel) + A["count_shrink"])
    else:
        assess_strength = 0.0

    # -- yoe_fit / hopper / only_consulting / months_since_ic -----------------
    yoe = float(intr["yoe"])
    peak_y = profile.role.peak_years; sigma_y = profile.role.sigma_years
    yoe_fit = float(np.exp(-((yoe - peak_y) ** 2) / (2 * sigma_y ** 2)))

    avg_tenure = float(intr["avg_tenure_months"])
    n_jobs_i = float(intr["n_jobs"])
    HD = method.hopper_def
    hopper = float(n_jobs_i >= HD["min_jobs"] and avg_tenure < HD["mean_tenure_below_months"])

    LEX = method.lexicons
    CONSULT = LEX["consulting"]; PRODIND = LEX["product_industries"]
    ICTOK = LEX["ic_tokens"]; MGTOK = LEX["mgmt_tokens"]

    def _is_ic(tl):
        return any(k in tl for k in ICTOK) and not any(k in tl for k in MGTOK)

    n_cons = 0
    months_since_ic = 999.0
    for i, j in enumerate(job_metas):
        comp = (j.get("company") or "").lower()
        tl = (j.get("title") or "").lower()
        if any(x in comp for x in CONSULT):
            n_cons += 1
        if _is_ic(tl):
            months_since_ic = min(months_since_ic, float(msince[i]))
    only_consulting = float(n_jobs > 0 and n_cons == n_jobs)

    # -- location ladder (features.py L330-363) --------------------------------
    LL = method.location_ladder
    LS = LL["scores"]; LV4 = LL["v4"]
    loc = str(intr["location"])
    reloc = bool(intr["willing_to_relocate"])
    pref_cities = profile.locations.preferred
    ok_cities = profile.locations.acceptable
    reloc_ok = profile.locations.relocation_acceptable
    pref = any(x in loc for x in pref_cities)
    okc = any(x in loc for x in ok_cities)
    india = (any(x in loc for x in LL["india_markers"]) or pref or okc)
    eff_reloc = reloc and reloc_ok
    if pref:
        loc_fit2 = LS["preferred"]
    elif okc:
        loc_fit2 = LS["ok_city"]
    elif india and eff_reloc:
        loc_fit2 = LS["india_relocate"]
    elif india:
        loc_fit2 = LS["india_no_reloc"]
    elif eff_reloc:
        loc_fit2 = LS["abroad_relocate"]
    else:
        loc_fit2 = LS["abroad_no_reloc"]

    remote_pref = bool(intr["remote_pref"])
    no_reloc = not reloc
    city_ok = pref or okc
    loc2 = (LV4["india_no_reloc_override"]
            if np.isclose(loc_fit2, LS["india_no_reloc"]) else float(loc_fit2))
    if remote_pref and no_reloc and not city_ok:
        loc2 = min(loc2, LV4["remote_noreloc_offcity_cap"])
    if remote_pref:
        loc2 *= LV4["remote_pref_damp"]

    # -- integrity ladder (features.py L365-394) --------------------------------
    IR = method.integrity
    max_role = float(intr["max_role_months"])
    integ = 1.0
    hard_conditions = [
        career_m > 0 and career_m / 12.0 > yoe + IR["career_sum_slack_years"],
        max_role / 12.0 > yoe + IR["single_role_slack_years"],
        career_m > 0 and yoe * 12 > career_m * IR["yoe_vs_history"]["ratio"]
                                    + IR["yoe_vs_history"]["slack_months"],
        summary_years is not None and abs(yoe - summary_years) > IR["summary_yoe_tolerance"],
        int(intr["n_expert_zero_dur"]) > 0,
        int(intr["n_expert"]) >= IR["too_many_expert"],
    ]
    for cond in hard_conditions:
        if cond:
            integ *= IR["hard"]
    soft_skill = career_m > 0 and n_skill_dur_exceed >= SD["min_count"]
    if soft_skill:
        integ *= 1.0 - min(SD["max_pen"], SD["per_skill_pen"] * n_skill_dur_exceed)
    if float(intr["salary_min"]) > float(intr["salary_max"]):
        integ *= IR["salary_inverted"]
    if int(intr["anach"]) == 1:
        integ *= IR["anachronism"]
    la, su = str(intr["last_active_date"]), str(intr["signup_date"])
    la_lt_signup = bool(la and su and la < su)
    if la_lt_signup:
        integ *= IR["la_lt_signup"]
    if int(intr["concurrent_deg"]) == 1:
        integ *= IR["concurrent_deg"]

    # -- availability (features.py L397-415) ------------------------------------
    AV = method.availability; AW = AV["weights"]
    la_date = _parse_date(la)
    mi = _months_between(la_date, REF) if la_date else AV["default_months_inactive"]
    mi = max(0.0, mi)
    rrr_raw = float(intr["recruiter_response_rate"])
    rrr = max(0.0, rrr_raw)
    rrr_d = 1.0 if rrr_raw < 0 else rrr_raw
    raw = (AW["recency"] * np.exp(-mi / AV["inactive_halflife_months"]) +
           AW["response"] * rrr +
           AW["open_to_work"] * float(intr["open_to_work_flag"]) +
           AW["interview"] * float(intr["interview_completion_rate"]) +
           AW["completeness"] * float(intr["profile_completeness_score"]) / 100.0)
    avail_mult = AV["base"] + AV["span"] * raw
    DM = AV["dormancy"]
    dormant = mi > DM["months_inactive"] and rrr_d < DM["rrr_below"]
    low_rr = rrr_d < DM["rrr_below"]
    availability = avail_mult
    if dormant:
        availability *= DM["damp"]
    elif low_rr:
        availability *= AV["low_rr_only_damp"]

    # -- notice tiers (features.py L418-423) -------------------------------------
    days = float(intr["notice_period_days"])
    notice_pen = None
    for tier in method.notice_tiers:
        if days <= tier["max_days"]:
            notice_pen = tier["mult"]
            break
    if notice_pen is None:
        notice_pen = method.notice_tiers[-1]["mult"]

    # -- lexical proxy per signal -------------------------------------------
    candidate_text_all = head_text + " " + jt_all + " " + " ".join(names)
    lex_by_id = {s.id: _lexical_proxy(s.query, candidate_text_all) for s in profile.signals}
    lex_fit = float(np.mean(list(lex_by_id.values()))) if sig_ids else 0.0

    # -- dense_fit (rules.py L119-129) --------------------------------------
    dense_fit = 0.0
    for i, s in enumerate(profile.signals):
        dense_fit += s.dense_weight * float(recencywt[i])
    de = profile.dense_extras
    yoe_fit_w = float(de.get("yoe_fit_weight", 0.0))
    domain_ratio_w = float(de.get("domain_ratio_weight", 0.0))
    dense_fit += yoe_fit_w * yoe_fit + domain_ratio_w * domain_nlp_ratio

    # -- additive composite (rules.py L134-138) ------------------------------
    AWc = method.additive_weights
    fit_raw = AWc["dense"] * dense_fit + AWc["lexical"] * lex_fit + AWc["depth"] * depth_bonus

    # -- normalize dense_fit's max attainable value so `overall` has an
    #    absolute 0-1 anchor (see FitResult docstring). dense_fit maxes out at
    #    sum(dense_weight) + yoe_fit_weight + domain_ratio_weight (cosine sim
    #    and yoe_fit/domain_nlp_ratio are all naturally <= 1); lex_fit and
    #    depth_bonus are already <= 1.
    max_dense_fit = (sum(s.dense_weight for s in profile.signals)
                      + yoe_fit_w + domain_ratio_w)
    max_fit_raw = (AWc["dense"] * max_dense_fit + AWc["lexical"] * 1.0
                   + AWc["depth"] * 1.0)
    dense_component = fit_raw / max_fit_raw if max_fit_raw > 0 else 0.0
    dense_component = float(np.clip(dense_component, 0.0, 1.0))

    # -- (1) evidence gate ----------------------------------------------------
    EG = method.evidence_gate
    g_evid = EG["floor"] + EG["span"] * evid_coverage

    # -- (2) claim-consistency discount ----------------------------------------
    CC = method.claim_consistency
    m_claim = 1.0 if ai_claimed_n == 0 else CC["base"] + CC["span"] * ai_corr

    # -- (3) assessment bonus --------------------------------------------------
    AB = method.assessment_bonus
    assess_corr = assess_strength * min(1.0, evid_coverage / AB["full_credit_cov"])
    m_assess = 1.0 + AB["weight"] * assess_corr

    # -- (4) recency ladder (gated by stale_ic_role) ---------------------------
    recency_mult = 1.0
    if profile.red_flag_enabled("stale_ic_role"):
        for tier in sorted(method.recency_ladder, key=lambda t: t["gt"]):
            if months_since_ic > tier["gt"]:
                recency_mult = tier["mult"]

    # -- (5) experience band ----------------------------------------------------
    EB = method.experience_band
    m_exp_band = EB["base"] + EB["span"] * yoe_fit

    # -- red-flag damps (gated per-JD) ------------------------------------------
    D = method.damps
    red_flags_fired = []
    m_cv = 1.0
    if profile.red_flag_enabled("cv_primary") and cv_primary == 1.0:
        m_cv = D["cv_primary"]
        red_flags_fired.append("cv_primary")
    m_hopper = 1.0
    if profile.red_flag_enabled("job_hopper") and hopper == 1.0:
        m_hopper = D["hopper"]
        red_flags_fired.append("job_hopper")
    m_only_consult = 1.0
    if profile.red_flag_enabled("only_consulting"):
        m_only_consult = 1.0 - D["only_consulting"] * only_consulting
        if only_consulting == 1.0:
            red_flags_fired.append("only_consulting")
    if profile.red_flag_enabled("stale_ic_role") and recency_mult < 1.0:
        red_flags_fired.append("stale_ic_role")

    # -- location damp (always-on geometry) --------------------------------------
    m_loc = D["loc_base"] + D["loc_span"] * loc2
    if loc2 <= D["loc_floor_threshold"]:
        m_loc *= D["loc_floor_damp"]

    gate_mult = (g_evid * m_claim * m_assess * recency_mult * m_exp_band
                 * m_cv * m_hopper * m_only_consult * m_loc
                 * integ * availability * notice_pen)

    overall = 100.0 * dense_component * gate_mult
    overall = float(np.clip(overall, 0.0, 100.0))

    # -- per-signal scorecard --------------------------------------------------
    per_signal = []
    for i, s in enumerate(profile.signals):
        contrib = s.dense_weight * float(recencywt[i])
        per_signal.append(SignalScore(
            id=s.id, label=s.label, dense=float(recencywt[i]),
            lexical=float(lex_by_id[s.id]), evidence=float(evid_by_id[s.id]),
            weight=float(s.dense_weight), contribution=float(contrib),
        ))

    # -- gates report ------------------------------------------------------------
    gates = [
        GateResult("integrity", float(integ), bool(integ < 0.999)),
        GateResult("availability", float(availability), bool(availability < 0.999)),
        GateResult("notice", float(notice_pen), bool(notice_pen < 0.999)),
        GateResult("location", float(m_loc), bool(m_loc < 0.999)),
    ]

    # -- gaps: coaching strings ---------------------------------------------------
    gaps = _build_gaps(profile, per_signal, gates, red_flags_fired,
                       days=days, notice_pref_days=profile.role.notice_preference_days,
                       dormant=dormant, low_rr=low_rr, months_since_ic=months_since_ic)

    return FitResult(
        overall=overall, per_signal=per_signal, gates=gates,
        red_flags=red_flags_fired, gaps=gaps,
        candidate_id=str(cid), role_title=profile.role.title,
    )


# ---------------------------------------------------------------------------
# gap / coaching text
# ---------------------------------------------------------------------------
_LOW_SIGNAL_THRESHOLD = 0.35


def _build_gaps(profile, per_signal, gates, red_flags_fired, *,
                days, notice_pref_days, dormant, low_rr, months_since_ic) -> list:
    gaps = []

    for sig in per_signal:
        if sig.dense < _LOW_SIGNAL_THRESHOLD and sig.evidence < _LOW_SIGNAL_THRESHOLD:
            gaps.append(
                f"Weak evidence for \"{sig.label}\" — add specific, quantified "
                f"career-history bullets that demonstrate this (current dense "
                f"similarity {sig.dense:.2f}, evidence {sig.evidence:.2f}).")

    for g in gates:
        if not g.damped:
            continue
        if g.name == "integrity":
            gaps.append(
                "Profile has an internal consistency issue (tenure/YoE/salary/"
                "certification mismatch) that is damping fit — review career "
                "history and years_of_experience for accuracy.")
        elif g.name == "availability":
            reason = []
            if dormant:
                reason.append("inactive too long with a low recruiter response rate")
            elif low_rr:
                reason.append("low recruiter response rate")
            reason_txt = " and ".join(reason) or "low platform-activity signals"
            gaps.append(
                f"Availability is damping fit ({reason_txt}) — respond to "
                "recruiter outreach and keep the profile active.")
        elif g.name == "notice":
            gaps.append(
                f"Notice period ({int(days)} days) is longer than this role's "
                f"preference (~{notice_pref_days} days) — a shorter notice "
                "period would remove this damp.")
        elif g.name == "location":
            gaps.append(
                "Location fit is damping the score — this role prefers "
                "specific cities/relocation/remote terms not currently matched "
                "by the candidate's stated location and relocation preference.")

    if "cv_primary" in red_flags_fired:
        gaps.append(
            "Career history reads as primarily out-of-domain relative to this "
            "role's domain — reframe accomplishments toward the JD's domain "
            "terms where genuinely applicable.")
    if "job_hopper" in red_flags_fired:
        gaps.append(
            "Job-hopper pattern detected (short average tenure across many "
            "roles) — this JD flags frequent short stints; longer, more "
            "recent tenure would help.")
    if "only_consulting" in red_flags_fired:
        gaps.append(
            "Career history is entirely consulting/services-firm roles — this "
            "JD wants direct product-company depth; highlight any embedded/"
            "client-owned delivery work.")
    if "stale_ic_role" in red_flags_fired:
        gaps.append(
            f"It has been ~{months_since_ic:.0f} months since the candidate's "
            "most recent individual-contributor (hands-on) role — this is a "
            "hands-on role, so recent hands-on work would strengthen fit.")

    return gaps
