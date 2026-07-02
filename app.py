"""Resume-Fit — JD → profile compiler (Streamlit HF Space, v1 / Module 1).

Paste a JD + your Google AI Studio key → a schema-valid jd_profile.yaml (drives
the RedRob ranking engine) + a coaching jd_meta.yaml, produced by the depth-1 RLM
harness. BYO key: held only in session state, never stored server-side or logged.
"""
from __future__ import annotations

import io
import json
import os

import streamlit as st
import yaml
from dotenv import load_dotenv

from harness.backends import (GEMINI_MODELS, GEMMA_MODELS, DEFAULT_MODEL,
                              GoogleGenAIBackend, RateLimitError)
from harness.coerce import compile_jd, to_yaml
from harness.logging_utils import HarnessLogger
from harness.validate import validate_profile_dict

load_dotenv()

st.set_page_config(page_title="Resume-Fit — JD Compiler", page_icon="🧭", layout="wide")


def extract_text(upload) -> str:
    name = (upload.name or "").lower()
    data = upload.read()
    if name.endswith(".pdf"):
        import fitz  # pymupdf
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return data.decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# sidebar — BYO key + model
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

st.title("🧭 Resume-Fit — JD → profile compiler")
st.write("Turn any job description into a **schema-valid `jd_profile.yaml`** "
         "(for the RedRob ranking engine) + a **coaching `jd_meta.yaml`**.")

# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
col_in, col_opt = st.columns([3, 1])
with col_opt:
    up = st.file_uploader("…or upload a JD", type=["pdf", "docx", "txt"])
with col_in:
    seed = extract_text(up) if up else ""
    jd_text = st.text_area("Paste the job description", value=seed, height=280,
                           placeholder="Paste the full JD prose here…")

go = st.button("Compile JD", type="primary", disabled=not (jd_text.strip() and key))
if not key:
    st.info("Enter your Google AI Studio API key in the sidebar to compile.")

# --------------------------------------------------------------------------- #
# compile
# --------------------------------------------------------------------------- #
if go:
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
            header.caption(f"**{len(events)}/≈14 calls** · {int(sum_tok)} tok · "
                           f"{sum_el:.1f}s")
            lines.markdown("\n".join(f"- {r}" for r in rows))

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

# --------------------------------------------------------------------------- #
# results (persist across reruns via session_state)
# --------------------------------------------------------------------------- #
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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["jd_profile.yaml (editable)", "jd_meta.yaml (coaching)", "health", "telemetry"])

    with tab1:
        st.caption("Edit and re-validate before download — the human-in-the-loop backstop.")
        edited = st.text_area("jd_profile.yaml", value=st.session_state["res_profile_yaml"],
                              height=460, key="edit_profile")
        c1, c2 = st.columns(2)
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

    with tab2:
        st.code(st.session_state["res_meta_yaml"], language="yaml")
        st.download_button("Download jd_meta.yaml", st.session_state["res_meta_yaml"],
                           file_name="jd_meta.yaml")

    with tab3:
        st.json(health)

    with tab4:
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
                               file_name="harness_run.jsonl", mime="application/x-ndjson")
