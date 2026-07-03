"""Resume-Fit — JD → profile compiler + Résumé → fit scorer (Streamlit HF Space).

Two flows in one app:
  1. JD → profile (Module 1, unchanged): paste a JD + your Google AI Studio key →
     a schema-valid jd_profile.yaml + coaching jd_meta.yaml, via a depth-1 RLM
     harness.
  2. Résumé → fit (Module 2 / Part C4): upload/paste a résumé → compile_resume
     (RLM harness, live telemetry) → an editable HITL form → approve → score
     against a JD profile (compiled in this session, or a gold eval JD) →
     an interpretable fit scorecard + gap analysis.

BYO key: held only in session state, never stored server-side or logged.
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from harness.backends import (GEMINI_MODELS, GEMMA_MODELS, DEFAULT_MODEL,
                              GoogleGenAIBackend, RateLimitError)
from harness.candidate_fields import validate_candidate
from harness.coerce import compile_jd
from harness.ingest import OCR_MAX_PAGES, extract_text  # shared file→text seam (JD + résumé)
from harness.logging_utils import HarnessLogger
from harness.resume import compile_resume
from harness.validate import validate_profile_dict
from redrob_ranker import profile as rprofile
from redrob_ranker.fit import score_candidate
from store import get_store
from store.schema import (CandidateRecordRow, CorrectionRow, FitRunRow,
                          ProfileRow, ResumeRow)
from eval_study.compare import discover_dataset, run_study, write_report, REPORT_DIR

load_dotenv()

st.set_page_config(page_title="Resume-Fit", page_icon="🧭", layout="wide")

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_JD_DIR = os.path.join(_HERE, "data", "eval_jds")
METHOD_PATH = os.path.join(_HERE, "jd", "method_config.yaml")
EVAL_RUNS_DIR = os.path.join(REPORT_DIR, "runs")
_METHOD_LABEL_SHORT = {
    "A_our_engine": "A) Our engine",
    "B_rubric_llm": "B) Rubric LLM",
    "C_naive_llm": "C) Naive LLM",
}


# --------------------------------------------------------------------------- #
# workspace / store bootstrap — one MemoryStore per session
# --------------------------------------------------------------------------- #
def _workspace_id() -> str:
    if "ws_id" not in st.session_state:
        st.session_state["ws_id"] = f"ws_{uuid.uuid4().hex[:12]}"
    return st.session_state["ws_id"]


def _get_workspace():
    if "store" not in st.session_state:
        st.session_state["store"] = get_store()   # default MemoryStore, session-only
        from store.schema import Workspace as WorkspaceRow
        st.session_state["store"].save_workspace(WorkspaceRow(
            workspace_id=_workspace_id(), name="Session workspace",
            created_at=_now_iso()))
    return st.session_state["store"]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def _diff_dict(before: dict, after: dict, prefix: str = "") -> list:
    """Flat list of (field_path, before, after) for values that changed.

    Shallow-recurses into nested dicts; lists are compared as whole values
    (career_history/education/skills are edited as whole blocks in the HITL
    table editors, so a list-level diff is the meaningful unit)."""
    out = []
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        path = f"{prefix}.{k}" if prefix else k
        bv, av = before.get(k), after.get(k)
        if isinstance(bv, dict) and isinstance(av, dict):
            out.extend(_diff_dict(bv, av, path))
        elif bv != av:
            out.append((path, bv, av))
    return out


def _live_status_events(events: list, header_ph, lines_ph, total_hint: str = "?") -> None:
    """Render accumulated leaf-call events into an st.status panel (mirrors the
    JD flow's live telemetry rendering)."""
    rows = []
    sum_tok = sum_el = 0.0
    for e in events:
        tok = e.get("tokens") or {}
        it, ot, tt = (tok.get("prompt_tokens"), tok.get("output_tokens"),
                      tok.get("total_tokens"))
        rows.append(f"{'✓' if e['ok'] else '✗'} {e['leaf']} · "
                    f"{e['elapsed_s']:.1f}s · {it}/{ot}/{tt} tok")
        sum_tok += tt or 0
        sum_el += e["elapsed_s"]
    header_ph.caption(f"**{len(events)}/≈{total_hint} calls** · {int(sum_tok)} tok · "
                      f"{sum_el:.1f}s")
    lines_ph.markdown("\n".join(f"- {r}" for r in rows))


CAREER_COLS = ["company", "title", "start_date", "end_date", "duration_months",
              "is_current", "industry", "company_size", "description"]
EDUCATION_COLS = ["institution", "degree", "field_of_study", "start_year",
                  "end_year", "grade", "tier"]
SKILL_COLS = ["name", "proficiency", "endorsements", "duration_months"]


def _career_history_to_df(rows: list) -> "pd.DataFrame":
    if not rows:
        return pd.DataFrame(columns=CAREER_COLS)
    return pd.DataFrame([{k: r.get(k) for k in CAREER_COLS} for r in rows])


def _df_to_career_history(df: "pd.DataFrame") -> list:
    out = []
    for _, row in df.iterrows():
        company = str(row.get("company") or "").strip()
        title = str(row.get("title") or "").strip()
        if not company and not title:
            continue   # skip fully-blank editor rows
        end_date = row.get("end_date")
        end_date = None if (end_date is None or str(end_date).strip() == "" or
                            str(end_date).lower() == "none") else str(end_date).strip()
        dur = row.get("duration_months")
        try:
            dur = int(dur) if dur is not None and str(dur).strip() != "" else 0
        except (TypeError, ValueError):
            dur = 0
        out.append({
            "company": company or "Unknown Company",
            "title": title or "Unknown Title",
            "start_date": str(row.get("start_date") or "").strip(),
            "end_date": end_date,
            "duration_months": max(0, dur),
            "is_current": bool(row.get("is_current")),
            "industry": str(row.get("industry") or "Technology").strip(),
            "company_size": str(row.get("company_size") or "201-500").strip(),
            "description": str(row.get("description") or "").strip(),
        })
    return out


def _education_to_df(rows: list) -> "pd.DataFrame":
    if not rows:
        return pd.DataFrame(columns=EDUCATION_COLS)
    return pd.DataFrame([{k: r.get(k) for k in EDUCATION_COLS} for r in rows])


def _df_to_education(df: "pd.DataFrame") -> list:
    out = []
    for _, row in df.iterrows():
        institution = str(row.get("institution") or "").strip()
        degree = str(row.get("degree") or "").strip()
        if not institution and not degree:
            continue
        entry = {
            "institution": institution or "Unknown Institution",
            "degree": degree or "Bachelor's",
            "field_of_study": str(row.get("field_of_study") or "General Studies").strip(),
            "start_year": int(row["start_year"]) if pd.notna(row.get("start_year")) else 2015,
            "end_year": int(row["end_year"]) if pd.notna(row.get("end_year")) else 2019,
        }
        grade = row.get("grade")
        if grade is not None and str(grade).strip():
            entry["grade"] = str(grade).strip()
        tier = row.get("tier")
        entry["tier"] = tier if tier in ("tier_1", "tier_2", "tier_3", "tier_4", "unknown") \
            else "unknown"
        out.append(entry)
    return out


def _skills_to_df(rows: list) -> "pd.DataFrame":
    if not rows:
        return pd.DataFrame(columns=SKILL_COLS)
    return pd.DataFrame([{k: r.get(k) for k in SKILL_COLS} for r in rows])


def _df_to_skills(df: "pd.DataFrame") -> list:
    out = []
    for _, row in df.iterrows():
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        prof = row.get("proficiency")
        prof = prof if prof in ("beginner", "intermediate", "advanced", "expert") \
            else "intermediate"
        endorsements = row.get("endorsements")
        try:
            endorsements = int(endorsements) if pd.notna(endorsements) else 0
        except (TypeError, ValueError):
            endorsements = 0
        entry = {"name": name, "proficiency": prof, "endorsements": max(0, endorsements)}
        dur = row.get("duration_months")
        if pd.notna(dur):
            try:
                entry["duration_months"] = max(0, int(dur))
            except (TypeError, ValueError):
                pass
        out.append(entry)
    return out


def _list_gold_jds() -> list:
    """[(slug, path_to_jd_profile.yaml), ...] for every gold JD with a profile."""
    out = []
    if not os.path.isdir(GOLD_JD_DIR):
        return out
    for slug in sorted(os.listdir(GOLD_JD_DIR)):
        p = os.path.join(GOLD_JD_DIR, slug, "jd_profile.yaml")
        if os.path.isfile(p):
            out.append((slug, p))
    return out


def _render_scorecard(fr: dict) -> None:
    """Render a fit scorecard dict (FitResult.to_dict()) — shared by the single
    and batch Résumé → fit paths so the rendering logic lives in one place."""
    st.metric("Overall fit index", f"{fr['overall']:.1f} / 100")
    st.caption(
        "This is an **interpretable fit index** — the normalized dense/"
        "lexical/depth composite times every gate multiplier — "
        "**not a percentile** and not comparable across different JDs or "
        "candidate pools. See `redrob_ranker/fit.py`'s `FitResult` "
        "docstring for the exact formula.")

    st.markdown("**Per-signal contribution**")
    for sig in fr["per_signal"]:
        st.write(f"`{sig['id']}` — {sig['label']} "
                f"(weight {sig['weight']:.2f})")
        bc1, bc2 = st.columns([3, 1])
        with bc1:
            st.progress(min(1.0, max(0.0, sig["dense"])),
                       text=f"dense {sig['dense']:.2f} · "
                            f"lexical {sig['lexical']:.2f} · "
                            f"evidence {sig['evidence']:.2f}")
        with bc2:
            st.caption(f"contrib {sig['contribution']:.3f}")

    st.markdown("**Gates**")
    gcols = st.columns(len(fr["gates"]) or 1)
    for gcol, g in zip(gcols, fr["gates"]):
        label = f"{g['name']} {'⚠️' if g['damped'] else '✅'}"
        gcol.metric(label, f"{g['value']:.2f}")

    if fr["red_flags"]:
        st.markdown("**Red flags fired**")
        for rfg in fr["red_flags"]:
            st.warning(rfg)
    else:
        st.caption("No enabled red flags fired for this JD.")

    st.markdown("**Gap analysis / coaching**")
    if fr["gaps"]:
        for gtxt in fr["gaps"]:
            st.write(f"- {gtxt}")
    else:
        st.caption("No coaching gaps surfaced — strong match on every "
                  "measured channel and gate.")

    st.download_button(
        "Download scorecard (JSON)", json.dumps(fr, indent=2, ensure_ascii=False),
        file_name="fit_scorecard.json", mime="application/json",
        key=f"dl_scorecard_{fr.get('candidate_id', '')}_{id(fr)}")


def _load_profile_method_from_yaml(profile_yaml_text: str):
    """Load Profile+Method objects from an in-session jd_profile.yaml STRING, by
    writing it to a temp file — mirrors harness/validate.py:validate_profile_dict's
    temp-file + rprofile.load pattern (reused here rather than duplicated)."""
    d = yaml.safe_load(profile_yaml_text)
    import tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(d, tmp, sort_keys=False, allow_unicode=True)
        tmp.close()
        return rprofile.load(tmp.name, METHOD_PATH)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _pick_jd_for_batch() -> "tuple[str, str] | None":
    """Resolve the JD picked in the (shared) JD selectbox to
    (jd_label, profile_yaml_text), or None if nothing is available yet.
    Mirrors tab_fit step 3's JD resolution, reused by batch mode."""
    jd_choice = st.session_state.get("jd_choice_batch")
    if not jd_choice:
        return None
    if jd_choice == "This session's compiled JD":
        return "session-compiled JD", st.session_state.get("session_jd_profile_yaml", "")
    gold = dict(_list_gold_jds())
    slug = jd_choice[len("Gold: "):]
    path = gold.get(slug)
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return slug, fh.read()


def _render_batch_fit_tab(files: list) -> None:
    """Batch mode: parse + score every uploaded résumé against the JD picked
    below, then render a sorted leaderboard. Auto-approves sentinels (no
    per-field HITL form here — single-résumé mode above is where HITL
    correction lives) and persists each as a fit_run (+resume/candidate rows)
    to the session store. A failed file is marked errored and does not abort
    the rest of the batch."""
    st.caption(f"**Batch mode** — {len(files)} résumés uploaded. Sentinels are "
              "auto-approved here (no per-field review); use single-résumé mode "
              "above for human-in-the-loop correction.")

    jd_options = []
    if "session_jd_profile_yaml" in st.session_state:
        jd_options.append("This session's compiled JD")
    gold = _list_gold_jds()
    jd_options.extend(f"Gold: {slug}" for slug, _ in gold)

    if not jd_options:
        st.info("No JD available yet. Compile one in the **JD → profile** tab "
                "and click \"Use in Résumé → fit tab\", or rely on the gold "
                "JDs in `data/eval_jds/` (none found on disk).")
        return

    st.selectbox("JD profile", jd_options, key="jd_choice_batch")
    run_batch = st.button("Score batch", type="primary", disabled=not key,
                          key="batch_go")
    if not key:
        st.info("Enter your Google AI Studio API key in the sidebar to score.")

    if run_batch:
        picked = _pick_jd_for_batch()
        if picked is None:
            st.error("Could not resolve the selected JD.")
            return
        jd_label, profile_yaml_text = picked
        try:
            jd_profile, jd_method = _load_profile_method_from_yaml(profile_yaml_text)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load the JD profile: {exc}")
            return

        try:
            backend = GoogleGenAIBackend(api_key=key, model=model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not initialize the model backend: {exc}")
            return

        try:
            ocr_fallback = GoogleGenAIBackend(api_key=key, model=GEMINI_MODELS[0])
        except Exception:  # noqa: BLE001
            ocr_fallback = None

        store = _get_workspace()
        wsid = _workspace_id()
        results = []   # list[dict]: name, file, status, fit_result(dict)|None, error
        progress = st.progress(0.0, text="Starting batch…")
        for i, f in enumerate(files):
            frac = i / max(1, len(files))
            progress.progress(frac, text=f"Processing {f.name} ({i + 1}/{len(files)})…")
            row = {"file": f.name, "name": f.name, "status": "ok",
                  "fit_result": None, "error": None}
            try:
                f.seek(0)
                text = extract_text(f, ocr_backend=backend, ocr_fallback_backend=ocr_fallback)
                if not text.strip():
                    raise ValueError("No extractable text (empty/unsupported file).")

                blog = HarnessLogger()
                rres = compile_resume(text, backend, logger=blog, max_repairs=max_repairs)
                cand = rres.candidate

                vr = validate_candidate(cand)
                if not vr.ok:
                    raise ValueError(f"Candidate failed validation: {vr.error}")

                fit_result = score_candidate(cand, jd_profile, jd_method, backend,
                                             ref_date=jd_method.ref_date)

                row["name"] = cand.get("profile", {}).get("anonymized_name") or f.name
                row["fit_result"] = fit_result.to_dict()

                # persist: resume + candidate_record + fit_run (sentinels
                # auto-approved — no HITL correction rows in batch mode)
                resume_id = _new_id("resume")
                store.save_resume(ResumeRow(
                    resume_id=resume_id, workspace_id=wsid, name=row["name"],
                    raw_text=text, candidate_json=cand, created_at=_now_iso()))
                record_id = _new_id("rec")
                store.save_candidate_record(CandidateRecordRow(
                    record_id=record_id, workspace_id=wsid, resume_id=resume_id,
                    candidate_json=cand, created_at=_now_iso()))
                profile_id = _new_id("profile")
                store.save_profile(ProfileRow(
                    profile_id=profile_id, workspace_id=wsid, name=jd_label,
                    profile_yaml=profile_yaml_text, created_at=_now_iso()))
                store.save_fit_run(FitRunRow(
                    run_id=_new_id("run"), workspace_id=wsid, profile_id=profile_id,
                    record_id=record_id, result_json=fit_result.to_dict(),
                    overall=fit_result.overall, created_at=_now_iso()))
            except RateLimitError as exc:
                row["status"] = "error"
                row["error"] = "Rate-limited (free-tier cap)"
                st.warning(
                    f"**{f.name}**: hit the Google AI Studio free-tier rate cap "
                    "(HTTP 429 / RESOURCE_EXHAUSTED) — skipped, continuing with "
                    "the rest of the batch.")
            except Exception as exc:  # noqa: BLE001 — one bad file must not sink the batch
                row["status"] = "error"
                row["error"] = str(exc)
            results.append(row)
        progress.progress(1.0, text="Batch complete.")

        st.session_state["batch_results"] = results
        st.session_state["batch_jd_label"] = jd_label

    results = st.session_state.get("batch_results")
    if not results:
        return

    st.divider()
    st.subheader(f"Leaderboard vs. {st.session_state.get('batch_jd_label', 'JD')}")

    scored = [r for r in results if r["status"] == "ok"]
    errored = [r for r in results if r["status"] != "ok"]
    scored.sort(key=lambda r: r["fit_result"]["overall"], reverse=True)

    board_rows = []
    for r in scored:
        fr = r["fit_result"]
        top_flags = ", ".join(fr["red_flags"]) if fr["red_flags"] else "—"
        top_gap = fr["gaps"][0] if fr["gaps"] else "—"
        board_rows.append({
            "candidate": r["name"], "file": r["file"],
            "overall": round(fr["overall"], 1), "red_flags": top_flags,
            "top_gap": top_gap,
        })
    if board_rows:
        st.dataframe(pd.DataFrame(board_rows), use_container_width=True,
                    hide_index=True)
    else:
        st.caption("No résumés scored successfully.")

    if errored:
        st.markdown("**Errored files**")
        for r in errored:
            st.warning(f"**{r['file']}** — {r['error']}")

    for r in scored:
        with st.expander(f"{r['name']} ({r['file']}) — overall "
                         f"{r['fit_result']['overall']:.1f}"):
            _render_scorecard(r["fit_result"])


# --------------------------------------------------------------------------- #
# Dashboard tab helpers — read persisted eval_study artifacts + list run history
# --------------------------------------------------------------------------- #
def _read_scores_csv(path: str) -> "pd.DataFrame | None":
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _summary_from_scores(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rebuild the per-method summary table (pairwise accuracy / stability /
    tokens / latency) straight from a scores.csv DataFrame — avoids re-parsing
    the markdown report for numbers the Dashboard needs to chart."""
    from eval_study.compare import pairwise_accuracy, kendall_tau, stability_stddev

    rows = []
    for method, mdf in df.groupby("method"):
        accs, taus = [], []
        for jd_slug, jdf in mdf.groupby("jd_slug"):
            tier_scores = jdf.groupby("tier")["score"].mean().to_dict()
            acc = pairwise_accuracy(tier_scores)
            tau = kendall_tau(tier_scores)
            if acc is not None:
                accs.append(acc)
            if tau is not None:
                taus.append(tau)
        devs = []
        for (jd_slug, tier), gdf in mdf.groupby(["jd_slug", "tier"]):
            if len(gdf) >= 2:
                devs.append(stability_stddev(gdf["score"].tolist()))
        total_tokens = mdf["tokens"].dropna().sum() if "tokens" in mdf else 0
        rows.append({
            "method": _METHOD_LABEL_SHORT.get(method, method),
            "pairwise_accuracy": (sum(accs) / len(accs)) if accs else None,
            "kendall_tau": (sum(taus) / len(taus)) if taus else None,
            "stability_stddev": (sum(devs) / len(devs)) if devs else None,
            "total_tokens": total_tokens if total_tokens else None,
            "total_elapsed_s": round(mdf["elapsed_s"].sum(), 3),
            "n_calls": len(mdf), "n_errors": int(mdf["error"].notna().sum()
                                                 if "error" in mdf else 0),
        })
    return pd.DataFrame(rows)


def _list_eval_runs() -> list:
    """[(timestamp_dir_name, report_path, csv_path), ...] newest first."""
    out = []
    if not os.path.isdir(EVAL_RUNS_DIR):
        return out
    for name in sorted(os.listdir(EVAL_RUNS_DIR), reverse=True):
        d = os.path.join(EVAL_RUNS_DIR, name)
        report = os.path.join(d, "comparison_report.md")
        csvp = os.path.join(d, "scores.csv")
        if os.path.isdir(d) and os.path.isfile(csvp):
            out.append((name, report if os.path.isfile(report) else None, csvp))
    return out


# --------------------------------------------------------------------------- #
# sidebar — BYO key + model (shared by both flows) + workspace browser
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Setup")
    key = st.text_input("Google AI Studio API key", type="password",
                        value=os.environ.get("GOOGLE_API_KEY", ""),
                        help="Held only for this session — never stored or logged. "
                             "Get one free at aistudio.google.com.")
    model = st.selectbox("Model", GEMMA_MODELS + GEMINI_MODELS,
                        index=(GEMMA_MODELS + GEMINI_MODELS).index(DEFAULT_MODEL))
    max_repairs = st.slider("Max repair passes", 0, 4, 2)
    detail_depth = st.slider("Detail depth", 1, 7, 4,
                             help="Caps how many capability signals get a full detail "
                                  "call. Fewer = faster on the free tier (labels beyond "
                                  "the cap still become signals, just without an API call).")
    st.caption("Free within Google AI Studio limits, subject to their rate caps.")
    st.divider()
    st.caption("Harness = depth-1 RLM: focused per-field leaf extractions → "
               "validate via the engine's own `profile.load` → repair only the "
               "failing field. Never emits an invalid file.")

    st.divider()
    st.subheader("Workspace")
    ws = _get_workspace()
    wsid = _workspace_id()
    saved_profiles = ws.list_profiles(wsid)
    saved_resumes = ws.list_resumes(wsid)
    saved_runs = ws.list_fit_runs(wsid)
    st.caption(f"Session workspace `{wsid}`")
    with st.expander(f"Profiles ({len(saved_profiles)})", expanded=False):
        for p in reversed(saved_profiles):
            st.caption(f"**{p.name}** · {p.profile_id} · {p.created_at}")
    with st.expander(f"Résumés ({len(saved_resumes)})", expanded=False):
        for r in reversed(saved_resumes):
            st.caption(f"**{r.name}** · {r.resume_id} · {r.created_at}")
    with st.expander(f"Fit runs ({len(saved_runs)})", expanded=False):
        for fr in reversed(saved_runs):
            st.caption(f"overall **{fr.overall:.1f}** · {fr.run_id} · {fr.created_at}")

st.title("🧭 Resume-Fit")
st.write("Turn a job description into a **schema-valid `jd_profile.yaml`**, and/or "
         "score a résumé against a JD with an interpretable **fit scorecard + gap "
         "analysis** — human-in-the-loop throughout.")

tab_jd, tab_fit, tab_dash = st.tabs(["JD → profile", "Résumé → fit", "Dashboard"])


# =============================================================================
# TAB 1 — JD → profile (Module 1, unchanged behavior)
# =============================================================================
with tab_jd:
    col_in, col_opt = st.columns([3, 1])
    with col_opt:
        up = st.file_uploader("…or upload a JD", type=["pdf", "docx", "txt"], key="jd_up")
    with col_in:
        seed = extract_text(up) if up else ""
        jd_text = st.text_area("Paste the job description", value=seed, height=280,
                               placeholder="Paste the full JD prose here…", key="jd_text")

    go = st.button("Compile JD", type="primary",
                   disabled=not (jd_text.strip() and key), key="jd_go")
    if not key:
        st.info("Enter your Google AI Studio API key in the sidebar to compile.")

    # ------------------------------------------------------------------- #
    # compile
    # ------------------------------------------------------------------- #
    if go or st.session_state.pop("jd_retry_compile", False):
        try:
            backend = GoogleGenAIBackend(api_key=key, model=model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not initialize the model backend: {exc}")
            st.stop()

        log = HarnessLogger()
        events: list[dict] = []

        with st.status(f"Compiling with {model} (RLM leaf extractions)…",
                       expanded=True) as status:
            header = st.empty()
            lines = st.empty()

            def _on_event(entry) -> None:
                # compile_jd runs synchronously on Streamlit's script thread, so
                # writing to these placeholders renders each leaf as it completes.
                d = entry.as_dict() if hasattr(entry, "as_dict") else entry
                events.append(d)
                _live_status_events(events, header, lines, total_hint="14")

            try:
                res = compile_jd(jd_text, backend, max_repairs=max_repairs,
                                 logger=log, on_event=_on_event,
                                 max_signal_detail=detail_depth)
            except RateLimitError as exc:
                status.update(label="Hit the free-tier rate cap", state="error")
                st.session_state["res_rate_limited"] = str(exc)
                st.warning(
                    "Hit the Google AI Studio **free-tier rate cap** "
                    "(HTTP 429 / RESOURCE_EXHAUSTED). This is expected on free keys "
                    "under load — wait a minute and retry, or lower **Detail depth** / "
                    "**Max repair passes** in the sidebar to send fewer calls.")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                status.update(label="Compile failed", state="error")
                st.error(f"Compile failed: {exc}")
                st.stop()
            status.update(label="Compile complete", state="complete")

        st.session_state["res_profile_yaml"] = res.profile_yaml
        st.session_state["res_meta_yaml"] = res.meta_yaml
        st.session_state["res_health"] = res.health
        st.session_state["res_ok"] = res.validation.ok
        st.session_state["res_err"] = res.validation.error
        st.session_state["res_telemetry"] = [e.as_dict() for e in log.entries]

    # ------------------------------------------------------------------- #
    # results (persist across reruns via session_state)
    # ------------------------------------------------------------------- #
    if "res_profile_yaml" in st.session_state:
        ok = st.session_state["res_ok"]
        health = st.session_state["res_health"]
        if ok:
            st.success(f"Valid jd_profile.yaml · model **{health['model']}** · "
                       f"repairs {health['repairs']} · "
                       f"defaulted {health['defaulted'] or '—'} · "
                       f"sentineled {health['sentineled'] or '—'}")
        else:
            st.warning(f"Emitted a file but validation flagged: {st.session_state['res_err']}")

        defaulted = health.get("defaulted") or []
        sentineled = health.get("sentineled") or []
        if defaulted or sentineled:
            msg = []
            if defaulted:
                msg.append(f"defaulted (leaf call failed): **{', '.join(defaulted)}**")
            if sentineled:
                msg.append(f"sentineled after repair attempts: **{', '.join(sentineled)}**")
            st.info(
                "Some fields fell back to defaults instead of a model answer — "
                "often a transient upstream error (e.g. Google-side 500/503) that "
                "clears on retry, not necessarily a problem with this JD. "
                + "; ".join(msg))
            if st.button("Retry compile", key="jd_retry_btn"):
                st.session_state["jd_retry_compile"] = True
                st.rerun()

        jd_tab1, jd_tab2, jd_tab3, jd_tab4 = st.tabs(
            ["jd_profile.yaml (editable)", "jd_meta.yaml (coaching)", "health", "telemetry"])

        with jd_tab1:
            st.caption("Edit and re-validate before download — the human-in-the-loop backstop.")
            edited = st.text_area("jd_profile.yaml", value=st.session_state["res_profile_yaml"],
                                  height=460, key="edit_profile")
            c1, c2, c3 = st.columns(3)
            if c1.button("Re-validate"):
                try:
                    d = yaml.safe_load(edited)
                    vr = validate_profile_dict(d)
                    if vr.ok:
                        st.success("OK — validates against jd_profile.schema.json")
                    else:
                        st.error(vr.error)
                except yaml.YAMLError as exc:
                    st.error(f"YAML parse error: {exc}")
            c2.download_button("Download jd_profile.yaml", edited, file_name="jd_profile.yaml")
            if c3.button("Use in Résumé → fit tab"):
                # Hands this compiled profile to the fit tab's JD picker without
                # re-running the compiler — validated on load, not blindly trusted.
                st.session_state["session_jd_profile_yaml"] = edited
                st.session_state["session_jd_meta_yaml"] = st.session_state.get("res_meta_yaml", "")
                st.success("Saved — pick \"This session's compiled JD\" in the Résumé → fit tab.")

        with jd_tab2:
            st.code(st.session_state["res_meta_yaml"], language="yaml")
            st.download_button("Download jd_meta.yaml", st.session_state["res_meta_yaml"],
                               file_name="jd_meta.yaml")

        with jd_tab3:
            st.json(health)

        with jd_tab4:
            entries = st.session_state.get("res_telemetry", [])
            if not entries:
                st.caption("No telemetry recorded for this run.")
            else:
                total_tok = sum((e.get("tokens") or {}).get("total_tokens") or 0
                                for e in entries)
                total_el = sum(e["elapsed_s"] for e in entries)
                st.caption(f"{len(entries)} leaf calls · {int(total_tok)} tokens · "
                           f"{total_el:.1f}s total. Session-only — the API key is never "
                           "recorded or shown.")
                for e in entries:
                    mark = "✓" if e["ok"] else "✗"
                    tok = e.get("tokens") or {}
                    with st.expander(f"{mark} {e['leaf']} · {e['elapsed_s']:.1f}s · "
                                     f"{tok.get('total_tokens')} tok"):
                        if e.get("error"):
                            st.error(e["error"])
                        st.markdown("**System instruction**")
                        st.code(e.get("system") or "(none)", language="text")
                        st.markdown("**Prompt sent**")
                        st.code(e.get("prompt_full") or "", language="text")
                        st.markdown("**Raw response**")
                        st.code(e.get("raw_response") or "(none)", language="text")
                        st.markdown("**Parsed**")
                        st.json(e.get("parsed"))
                        st.markdown("**Tokens**")
                        st.write(e.get("tokens") or "—")
                jsonl = "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
                st.download_button("Download run log (JSONL)", jsonl,
                                   file_name="harness_run.jsonl", mime="application/x-ndjson",
                                   key="jd_jsonl_dl")


# =============================================================================
# TAB 2 — Résumé → fit (Module 2, Part C4)
# =============================================================================
with tab_fit:
    st.subheader("1 — Résumé input")
    batch_files = st.file_uploader(
        "Upload résumés (2+ switches to batch mode — leaderboard across every "
        "uploaded résumé; upload exactly one, or paste text below, for the "
        "single-résumé HITL review flow)",
        type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_up_batch")
    is_batch = len(batch_files) >= 2

    if is_batch:
        _render_batch_fit_tab(batch_files)

    else:
        rcol_in, rcol_opt = st.columns([3, 1])
        with rcol_opt:
            rup = batch_files[0] if batch_files else st.file_uploader(
                "…or upload a résumé", type=["pdf", "docx", "txt"], key="resume_up")
        with rcol_in:
            rseed = ""
            if rup:
                plain = extract_text(rup)
                # Image-only / scanned PDF: pymupdf's text layer is ~empty. If we
                # have a key, OCR it now via the currently-selected (Gemma) backend,
                # with a single-shot gemini-3-flash fallback if that also comes up
                # empty. The résumé form below is the HITL backstop for any OCR gaps.
                looks_scanned = (rup.name or "").lower().endswith(".pdf") and \
                    len(plain.strip()) < 20 * 5
                if looks_scanned and key:
                    try:
                        ocr_backend = GoogleGenAIBackend(api_key=key, model=model)
                        ocr_fallback = GoogleGenAIBackend(api_key=key, model=GEMINI_MODELS[0])
                    except Exception:  # noqa: BLE001
                        ocr_backend = ocr_fallback = None
                    if ocr_backend is not None:
                        with st.spinner(f"Image-only PDF detected — OCR via {model}…"):
                            rup.seek(0)
                            ocr_text = extract_text(rup, ocr_backend=ocr_backend,
                                                    ocr_fallback_backend=ocr_fallback)
                        if ocr_text.strip():
                            plain = ocr_text
                            n_pages_hint = min(OCR_MAX_PAGES, 5)
                            st.caption(
                                f"Image-only PDF detected — OCR via **{model}**, "
                                f"5 passes/page (up to {n_pages_hint} pages); "
                                "verify/fix in the form below (HITL).")
                elif looks_scanned and not key:
                    st.info("This looks like an image-only/scanned PDF — enter your "
                            "API key in the sidebar to OCR it.")
                rseed = plain
            resume_text = st.text_area("Paste résumé text", value=rseed, height=260,
                                       placeholder="Paste the full résumé text here…",
                                       key="resume_text")

        parse_go = st.button("Parse résumé", type="primary",
                             disabled=not (resume_text.strip() and key), key="resume_go")
        if not key:
            st.info("Enter your Google AI Studio API key in the sidebar to parse.")

        if parse_go:
            try:
                backend = GoogleGenAIBackend(api_key=key, model=model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not initialize the model backend: {exc}")
                st.stop()

            rlog = HarnessLogger()
            revents: list[dict] = []

            with st.status(f"Parsing résumé with {model} (RLM leaf extractions)…",
                           expanded=True) as rstatus:
                rheader = st.empty()
                rlines = st.empty()

                def _on_resume_event(entry) -> None:
                    d = entry.as_dict() if hasattr(entry, "as_dict") else entry
                    revents.append(d)
                    _live_status_events(revents, rheader, rlines, total_hint="6")

                try:
                    rres = compile_resume(resume_text, backend, logger=rlog,
                                          on_event=_on_resume_event, max_repairs=max_repairs)
                except RateLimitError as exc:
                    rstatus.update(label="Hit the free-tier rate cap", state="error")
                    st.warning(
                        "Hit the Google AI Studio **free-tier rate cap** "
                        "(HTTP 429 / RESOURCE_EXHAUSTED). Wait a minute and retry, or "
                        "lower **Max repair passes** in the sidebar to send fewer calls.")
                    st.stop()
                except Exception as exc:  # noqa: BLE001
                    rstatus.update(label="Résumé parse failed", state="error")
                    st.error(f"Résumé parse failed: {exc}")
                    st.stop()
                rstatus.update(label="Résumé parse complete", state="complete")

            # parsed (pre-HITL) snapshot — kept immutable for the delta capture later
            st.session_state["fit_parsed_candidate"] = copy.deepcopy(rres.candidate)
            st.session_state["fit_candidate"] = copy.deepcopy(rres.candidate)
            st.session_state["fit_resume_text"] = resume_text
            st.session_state["fit_health"] = rres.health
            st.session_state["fit_ok"] = rres.validation.ok
            st.session_state["fit_err"] = rres.validation.error
            st.session_state["fit_telemetry"] = [e.as_dict() for e in rlog.entries]
            st.session_state["fit_approved"] = False

        # ----------------------------------------------------------------------- #
        # HITL form — only once a candidate has been parsed this session
        # ----------------------------------------------------------------------- #
        if "fit_candidate" in st.session_state:
            st.divider()
            st.subheader("2 — Review & correct (human-in-the-loop)")

            health = st.session_state["fit_health"]
            if st.session_state["fit_ok"]:
                st.success(f"Valid candidate · model **{health['model']}** · "
                           f"repairs {health['repairs']} · "
                           f"defaulted {health['defaulted'] or '—'} · "
                           f"sentineled {health['sentineled'] or '—'}")
            else:
                st.warning(f"Emitted a candidate but validation flagged: {st.session_state['fit_err']}")

            cand = st.session_state["fit_candidate"]

            with st.expander("Telemetry (résumé parse)", expanded=False):
                entries = st.session_state.get("fit_telemetry", [])
                if not entries:
                    st.caption("No telemetry recorded for this run.")
                else:
                    total_tok = sum((e.get("tokens") or {}).get("total_tokens") or 0
                                    for e in entries)
                    total_el = sum(e["elapsed_s"] for e in entries)
                    st.caption(f"{len(entries)} leaf calls · {int(total_tok)} tokens · "
                               f"{total_el:.1f}s total. The API key is never recorded.")
                    for e in entries:
                        mark = "✓" if e["ok"] else "✗"
                        tok = e.get("tokens") or {}
                        st.markdown(f"**{mark} {e['leaf']}** · {e['elapsed_s']:.1f}s · "
                                   f"{tok.get('total_tokens')} tok")
                        if e.get("error"):
                            st.error(e["error"])
                        st.code(e.get("prompt_full") or "", language="text")
                        st.json(e.get("parsed"))

            # -- profile block ---------------------------------------------------- #
            st.markdown("**Profile**")
            p = cand.get("profile") or {}
            pc1, pc2, pc3 = st.columns(3)
            p["anonymized_name"] = pc1.text_input("Name (anonymized)",
                                                  value=p.get("anonymized_name", ""))
            p["headline"] = pc2.text_input("Headline", value=p.get("headline", ""))
            p["location"] = pc3.text_input("Location", value=p.get("location", ""))
            pc4, pc5, pc6 = st.columns(3)
            p["country"] = pc4.text_input("Country", value=p.get("country", "India"))
            p["years_of_experience"] = pc5.number_input(
                "Years of experience", min_value=0.0, max_value=50.0,
                value=float(p.get("years_of_experience", 0.0)), step=0.5)
            size_opts = ["1-10", "11-50", "51-200", "201-500", "501-1000",
                        "1001-5000", "5001-10000", "10001+"]
            cur_size = p.get("current_company_size", "201-500")
            p["current_company_size"] = pc6.selectbox(
                "Current company size", size_opts,
                index=size_opts.index(cur_size) if cur_size in size_opts else 3)
            pc7, pc8, pc9 = st.columns(3)
            p["current_title"] = pc7.text_input("Current title", value=p.get("current_title", ""))
            p["current_company"] = pc8.text_input("Current company", value=p.get("current_company", ""))
            p["current_industry"] = pc9.text_input("Current industry", value=p.get("current_industry", ""))
            p["summary"] = st.text_area("Summary", value=p.get("summary", ""), height=140)
            cand["profile"] = p

            # -- career history (editable table) ---------------------------------- #
            st.markdown("**Career history**")
            ch_df = _career_history_to_df(cand.get("career_history") or [])
            ch_edited = st.data_editor(
                ch_df, num_rows="dynamic", use_container_width=True, key="ch_editor",
                column_config={
                    "is_current": st.column_config.CheckboxColumn("Current?"),
                    "start_date": st.column_config.TextColumn("Start (YYYY-MM-DD)"),
                    "end_date": st.column_config.TextColumn("End (YYYY-MM-DD, blank if current)"),
                    "duration_months": st.column_config.NumberColumn("Duration (months)", min_value=0),
                })
            cand["career_history"] = _df_to_career_history(ch_edited)

            # -- education (editable table) ---------------------------------------- #
            st.markdown("**Education**")
            ed_df = _education_to_df(cand.get("education") or [])
            ed_edited = st.data_editor(
                ed_df, num_rows="dynamic", use_container_width=True, key="ed_editor",
                column_config={
                    "start_year": st.column_config.NumberColumn("Start year", min_value=1970, max_value=2030),
                    "end_year": st.column_config.NumberColumn("End year", min_value=1970, max_value=2035),
                    "tier": st.column_config.SelectboxColumn(
                        "Tier", options=["tier_1", "tier_2", "tier_3", "tier_4", "unknown"]),
                })
            cand["education"] = _df_to_education(ed_edited)

            # -- skills (editable table) --------------------------------------------- #
            st.markdown("**Skills**")
            sk_df = _skills_to_df(cand.get("skills") or [])
            sk_edited = st.data_editor(
                sk_df, num_rows="dynamic", use_container_width=True, key="sk_editor",
                column_config={
                    "proficiency": st.column_config.SelectboxColumn(
                        "Proficiency", options=["beginner", "intermediate", "advanced", "expert"]),
                    "endorsements": st.column_config.NumberColumn("Endorsements", min_value=0),
                    "duration_months": st.column_config.NumberColumn("Duration (months)", min_value=0),
                })
            cand["skills"] = _df_to_skills(sk_edited)

            # -- redrob_signals a résumé can't supply: advisory weight sliders ------ #
            st.markdown("**Platform-activity signals** (a résumé can't supply these — "
                        "sentineled to neutral defaults; adjust if you have real data)")
            rs = cand.get("redrob_signals") or {}
            sig1, sig2, sig3 = st.columns(3)
            rs["notice_period_days"] = sig1.slider(
                "Notice period (days)", 0, 180, int(rs.get("notice_period_days", 60)))
            rs["recruiter_response_rate"] = sig2.slider(
                "Recruiter response rate (advisory)", 0.0, 1.0,
                float(rs.get("recruiter_response_rate", 0.5)))
            rs["interview_completion_rate"] = sig3.slider(
                "Interview completion rate (advisory)", 0.0, 1.0,
                float(rs.get("interview_completion_rate", 0.5)))
            sig4, sig5, sig6 = st.columns(3)
            rs["profile_completeness_score"] = sig4.slider(
                "Profile completeness (advisory)", 0, 100,
                int(rs.get("profile_completeness_score", 50)))
            rs["open_to_work_flag"] = sig5.checkbox(
                "Open to work", value=bool(rs.get("open_to_work_flag", True)))
            rs["willing_to_relocate"] = sig6.checkbox(
                "Willing to relocate", value=bool(rs.get("willing_to_relocate", True)))
            cand["redrob_signals"] = rs

            st.session_state["fit_candidate"] = cand

            # -- approve --------------------------------------------------------------- #
            approve = st.button("Approve corrections", type="primary", key="fit_approve")
            if approve:
                vr = validate_candidate(cand)
                st.session_state["fit_validation_ok"] = vr.ok
                st.session_state["fit_validation_err"] = vr.error
                if vr.ok:
                    st.session_state["fit_approved"] = True
                    store = _get_workspace()
                    wsid = _workspace_id()

                    # delta: parsed (pre-HITL) vs approved candidate, persisted as
                    # individual correction rows.
                    parsed = st.session_state.get("fit_parsed_candidate") or {}
                    deltas = _diff_dict(parsed, cand)
                    resume_id = _new_id("resume")
                    for field_path, before, after in deltas:
                        store.save_correction(CorrectionRow(
                            correction_id=_new_id("corr"), workspace_id=wsid,
                            resume_id=resume_id, field_path=field_path,
                            before=before, after=after,
                            note="parsed-vs-approved HITL delta", created_at=_now_iso()))

                    store.save_resume(ResumeRow(
                        resume_id=resume_id, workspace_id=wsid,
                        name=cand.get("profile", {}).get("anonymized_name", "Candidate"),
                        raw_text=st.session_state.get("fit_resume_text", ""),
                        candidate_json=cand, created_at=_now_iso()))
                    record_id = _new_id("rec")
                    store.save_candidate_record(CandidateRecordRow(
                        record_id=record_id, workspace_id=wsid, resume_id=resume_id,
                        candidate_json=cand, created_at=_now_iso()))
                    st.session_state["fit_resume_id"] = resume_id
                    st.session_state["fit_record_id"] = record_id
                    st.session_state["fit_n_corrections"] = len(deltas)
                    st.success(f"Approved and validated · {len(deltas)} correction(s) "
                              "captured vs. the parsed résumé · saved to workspace.")
                else:
                    st.error(f"Still invalid after your edits: {vr.error}")

            if st.session_state.get("fit_approved"):
                n = st.session_state.get("fit_n_corrections", 0)
                st.caption(f"Approved · {n} HITL correction(s) recorded in `store.corrections` "
                          f"· resume_id `{st.session_state.get('fit_resume_id')}`.")

            # ----------------------------------------------------------------------- #
            # 3 — pick the JD to score against
            # ----------------------------------------------------------------------- #
            st.divider()
            st.subheader("3 — Pick a JD to score against")

            jd_options = []
            if "session_jd_profile_yaml" in st.session_state:
                jd_options.append("This session's compiled JD")
            gold = _list_gold_jds()
            jd_options.extend(f"Gold: {slug}" for slug, _ in gold)

            # also offer JD profiles saved to the workspace (e.g. from a prior
            # "Use in Résumé → fit tab" click persisted across reruns, or future
            # multi-profile workflows)
            if not jd_options:
                st.info("No JD available yet. Compile one in the **JD → profile** tab "
                        "and click \"Use in Résumé → fit tab\", or rely on the gold "
                        "JDs in `data/eval_jds/` (none found on disk).")
            else:
                jd_choice = st.selectbox("JD profile", jd_options, key="jd_choice")

                score_go = st.button(
                    "Score fit", type="primary",
                    disabled=not (st.session_state.get("fit_approved") and key),
                    key="score_go")
                if not st.session_state.get("fit_approved"):
                    st.caption("Approve corrections above before scoring.")

                if score_go:
                    try:
                        if jd_choice == "This session's compiled JD":
                            profile_yaml_text = st.session_state["session_jd_profile_yaml"]
                            jd_label = "session-compiled JD"
                        else:
                            slug = jd_choice[len("Gold: "):]
                            path = dict(gold)[slug]
                            with open(path, "r", encoding="utf-8") as fh:
                                profile_yaml_text = fh.read()
                            jd_label = slug
                        jd_profile, jd_method = _load_profile_method_from_yaml(profile_yaml_text)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not load the JD profile: {exc}")
                        st.stop()

                    try:
                        fit_backend = GoogleGenAIBackend(api_key=key, model=model)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not initialize the model backend: {exc}")
                        st.stop()

                    with st.spinner(f"Scoring against **{jd_label}** (live embeddings)…"):
                        try:
                            fit_result = score_candidate(
                                st.session_state["fit_candidate"], jd_profile, jd_method,
                                fit_backend, ref_date=jd_method.ref_date)
                        except RateLimitError as exc:
                            st.warning(
                                "Hit the Google AI Studio **free-tier rate cap** while "
                                "computing embeddings (HTTP 429 / RESOURCE_EXHAUSTED). "
                                "Wait a minute and retry.")
                            st.stop()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Scoring failed: {exc}")
                            st.stop()

                    st.session_state["fit_result"] = fit_result.to_dict()
                    st.session_state["fit_jd_label"] = jd_label

                    # persist: the JD profile (if not already saved) + the fit_run
                    store = _get_workspace()
                    wsid = _workspace_id()
                    profile_id = _new_id("profile")
                    store.save_profile(ProfileRow(
                        profile_id=profile_id, workspace_id=wsid, name=jd_label,
                        profile_yaml=profile_yaml_text, created_at=_now_iso()))
                    store.save_fit_run(FitRunRow(
                        run_id=_new_id("run"), workspace_id=wsid, profile_id=profile_id,
                        record_id=st.session_state.get("fit_record_id", ""),
                        result_json=fit_result.to_dict(), overall=fit_result.overall,
                        created_at=_now_iso()))

            # ----------------------------------------------------------------------- #
            # 4 — scorecard
            # ----------------------------------------------------------------------- #
            if "fit_result" in st.session_state:
                st.divider()
                st.subheader(f"4 — Scorecard vs. {st.session_state.get('fit_jd_label', 'JD')}")
                _render_scorecard(st.session_state["fit_result"])


# =============================================================================
# TAB 3 — Dashboard (live A/B/C comparison + session telemetry)
# =============================================================================
with tab_dash:
    st.subheader("Live comparison — our engine vs. naive single-LLM baselines")
    st.caption(
        "Runs `eval_study.compare.run_study` over a BOUNDED sample of the "
        "synthetic, labeled `single_column` résumé set (via `discover_dataset`) "
        "so a click here can't rack up large API cost. Computed on synthetic "
        "labeled data — **not a live percentile**, a sanity check of ranking "
        "agreement with a known strong/borderline/weak ordering.")

    dash_dataset_all = discover_dataset()
    max_pairs = len(dash_dataset_all) or 1
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        n_pairs = st.number_input(
            "Number of (JD, résumé) pairs", min_value=1,
            max_value=max_pairs, value=min(6, max_pairs), step=1,
            help=f"{max_pairs} pairs available in data/synthetic_resumes/ "
                 "(single_column layout).")
    with dcol2:
        n_repeats = st.number_input("Repeats per pair per method", min_value=1,
                                    max_value=5, value=2, step=1)

    run_cmp = st.button("Run live comparison", type="primary", disabled=not key,
                        key="dash_run_cmp")
    if not key:
        st.info("Enter your Google AI Studio API key in the sidebar to run a "
                "live comparison.")

    if run_cmp:
        dataset = discover_dataset(limit=int(n_pairs))
        if not dataset:
            st.error("No (JD, résumé) pairs found under data/synthetic_resumes/.")
        else:
            try:
                cmp_backend = GoogleGenAIBackend(api_key=key, model=model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not initialize the model backend: {exc}")
                cmp_backend = None

            if cmp_backend is not None:
                total_calls = len(dataset) * int(n_repeats) * 3   # 3 methods
                prog = st.progress(0.0, text="Starting comparison study…")
                status_ph = st.empty()
                done = {"n": 0}
                partial_rows = []

                def _on_row(row) -> None:
                    done["n"] += 1
                    partial_rows.append(row)
                    frac = min(1.0, done["n"] / max(1, total_calls))
                    status = "ERR" if row.error else "ok"
                    prog.progress(frac, text=f"{done['n']}/{total_calls} calls — "
                                             f"[{status}] {row.jd_slug}/{row.tier} "
                                             f"{row.method} rep={row.repeat}")

                try:
                    study = run_study(dataset, cmp_backend, repeats=int(n_repeats),
                                      on_row=_on_row)
                    prog.progress(1.0, text="Comparison complete.")

                    report_path, csv_path = write_report(study)

                    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                    run_dir = os.path.join(EVAL_RUNS_DIR, ts)
                    os.makedirs(run_dir, exist_ok=True)
                    write_report(study, out_dir=run_dir)

                    st.session_state["dash_last_run_ts"] = ts
                    st.success(f"Comparison complete — {len(study.rows)} scored "
                              f"calls. Wrote `{report_path}`, `{csv_path}`, and a "
                              f"timestamped copy under `data/eval_study/runs/{ts}/`.")
                except RateLimitError as exc:
                    st.warning(
                        "Hit the Google AI Studio **free-tier rate cap** "
                        "(HTTP 429 / RESOURCE_EXHAUSTED) partway through the "
                        "comparison. Showing partial results collected so far.")
                    if partial_rows:
                        from eval_study.compare import StudyResult
                        partial_study = StudyResult(rows=partial_rows,
                                                    repeats=int(n_repeats),
                                                    pairs=dataset)
                        write_report(partial_study)
                        st.info(f"Partial results ({len(partial_rows)} calls) "
                               "written to the persisted report.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Comparison study failed: {exc}")
                    if partial_rows:
                        from eval_study.compare import StudyResult
                        partial_study = StudyResult(rows=partial_rows,
                                                    repeats=int(n_repeats),
                                                    pairs=dataset)
                        write_report(partial_study)
                        st.info(f"Partial results ({len(partial_rows)} calls) "
                               "written to the persisted report.")

    st.divider()
    st.subheader("Persisted results")

    scores_path = os.path.join(REPORT_DIR, "scores.csv")
    report_path = os.path.join(REPORT_DIR, "comparison_report.md")
    scores_df = _read_scores_csv(scores_path)

    if scores_df is None or scores_df.empty:
        st.info("No comparison results yet — click **Run live comparison** "
                "above (or run `python -m eval_study.compare` from a shell) "
                "to populate this dashboard.")
    else:
        summary_df = _summary_from_scores(scores_df)
        st.caption(
            "Computed on synthetic labeled data (`data/synthetic_resumes/`) — "
            "not a live percentile. Ranking accuracy measures agreement with "
            "the known strong/borderline/weak gold ordering per JD.")

        st.markdown("**Summary (A/B/C)**")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        chart_df = summary_df.set_index("method")[["pairwise_accuracy"]].rename(
            columns={"pairwise_accuracy": "ranking accuracy"})
        st.markdown("**Ranking accuracy per method**")
        st.bar_chart(chart_df)

        st.markdown("**Stability (score std-dev across repeats — lower is more consistent)**")
        stab_df = summary_df.set_index("method")[["stability_stddev"]].rename(
            columns={"stability_stddev": "stability (std-dev)"})
        st.bar_chart(stab_df)

        st.markdown("**Cost / latency**")
        cost_df = summary_df[["method", "total_tokens", "total_elapsed_s", "n_calls",
                              "n_errors"]]
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        if os.path.isfile(report_path):
            with st.expander("Full comparison_report.md", expanded=False):
                with open(report_path, "r", encoding="utf-8") as fh:
                    st.markdown(fh.read())

    prior_runs = _list_eval_runs()
    if prior_runs:
        st.markdown("**Prior runs** (`data/eval_study/runs/`)")
        for ts, rpath, cpath in prior_runs:
            st.caption(f"`{ts}`" + (f" · report available" if rpath else "")
                      + f" · `{os.path.relpath(cpath, _HERE)}`")

    st.divider()
    st.subheader("Session telemetry")
    ws_dash = _get_workspace()
    wsid_dash = _workspace_id()
    dash_runs = ws_dash.list_fit_runs(wsid_dash)
    dash_resumes = ws_dash.list_resumes(wsid_dash)
    dash_profiles = ws_dash.list_profiles(wsid_dash)
    dash_corrections = ws_dash.list_corrections(wsid_dash)

    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    tcol1.metric("Fit runs (this session)", len(dash_runs))
    tcol2.metric("Résumés parsed", len(dash_resumes))
    tcol3.metric("JD profiles used", len(dash_profiles))
    tcol4.metric("HITL corrections captured", len(dash_corrections))

    if dash_runs:
        avg_overall = sum(r.overall for r in dash_runs) / len(dash_runs)
        st.caption(f"Average `overall` fit index across this session's fit "
                  f"runs: **{avg_overall:.1f} / 100** (session workspace "
                  f"`{wsid_dash}` — not comparable across different JDs).")
    else:
        st.caption("No fit runs recorded yet in this session — score a résumé "
                  "in the **Résumé → fit** tab (single or batch) to populate "
                  "session telemetry.")
