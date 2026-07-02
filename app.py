"""Resume-Fit — JD → profile compiler (Streamlit HF Space, v1 / Module 1).

Paste a JD + your Google AI Studio key → a schema-valid jd_profile.yaml (drives
the RedRob ranking engine) + a coaching jd_meta.yaml, produced by the depth-1 RLM
harness. BYO key: held only in session state, never stored server-side or logged.
"""
from __future__ import annotations

import io
import os

import streamlit as st
import yaml
from dotenv import load_dotenv

from harness.backends import GEMINI_MODELS, GEMMA_MODELS, DEFAULT_MODEL, GoogleGenAIBackend
from harness.coerce import compile_jd, to_yaml
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
    with st.spinner(f"Compiling with {model} (RLM leaf extractions)…"):
        try:
            res = compile_jd(jd_text, backend, max_repairs=max_repairs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Compile failed: {exc}")
            st.stop()
    st.session_state["res_profile_yaml"] = res.profile_yaml
    st.session_state["res_meta_yaml"] = res.meta_yaml
    st.session_state["res_health"] = res.health
    st.session_state["res_ok"] = res.validation.ok
    st.session_state["res_err"] = res.validation.error

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

    tab1, tab2, tab3 = st.tabs(["jd_profile.yaml (editable)", "jd_meta.yaml (coaching)", "health"])

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
