"""Single file-ingestion seam: any document → plain text.

This is Layer 1 (format decoding) and the ONE place both flows decode a file —
the JD compiler (`coerce.py`) and the résumé compiler (`resume.py`) both call
`extract_text`. It is deliberately independent of the RLM harness: what differs
between a JD and a résumé is the *semantic* extraction downstream (Layer 2 — leaf
prompts + validator + sentinels), NOT the byte-level decoding, so decoding is
shared and the schema-specific harnesses stay separate.

Supported sources
-----------------
- Streamlit ``UploadedFile`` (has ``.name`` + ``.read()``)
- a filesystem path (``str`` / ``os.PathLike``)
- raw ``bytes`` (pass ``filename=`` so the format can be inferred)
- a Google Docs share URL (fetched via its plain-text export endpoint; the doc
  must be link-viewable)

Formats: ``.pdf`` (pymupdf), ``.docx`` (python-docx), plain text. New sources
(e.g. a Google Docs API client, an object store) plug in here without touching
either harness.

OCR fallback (image-only / scanned PDFs)
-----------------------------------------
pymupdf's ``page.get_text()`` returns ~nothing for a scanned/image-only PDF
(no text layer). When that happens and an ``ocr_backend`` is supplied,
``extract_text`` renders each page to PNG and asks a multimodal model to
transcribe it (see ``_ocr_pdf`` / ``OCR_PROMPT`` below). This keeps OCR out of
the byte-decoding hot path for the (overwhelmingly common) text-layer PDFs —
it only engages when pymupdf's normal extraction comes back empty.
"""
from __future__ import annotations

import io
import os
import re

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt", ".md", ".text")

_GDOC_RE = re.compile(r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)")

# -- OCR fallback tuning -------------------------------------------------- #
OCR_PROMPT = ("Transcribe ALL text in this image verbatim in natural reading "
             "order. Output plain text only, no commentary.")
OCR_MAX_PAGES = 5             # cap pages OCR'd, to bound API calls
OCR_DPI = 200                 # render resolution for get_pixmap
OCR_MIN_CHARS_PER_PAGE = 20   # below this avg (non-whitespace) chars/page -> treat as image-only


def extract_text(source, *, filename: str | None = None, ocr_backend=None,
                 ocr_fallback_backend=None) -> str:
    """Decode *source* to plain text.

    Parameters
    ----------
    source : Streamlit UploadedFile | path-like | bytes | Google Docs URL
    filename : optional explicit name, used to infer the format when *source*
        is raw bytes or a file object without a usable ``.name``.
    ocr_backend : optional Backend (e.g. Gemma) with `generate_multimodal`.
        When a PDF's pymupdf text layer is effectively empty, this backend is
        used to OCR the rendered page images. If omitted, an image-only PDF
        just returns whatever (near-empty) text pymupdf found.
    ocr_fallback_backend : optional Backend (e.g. Gemini) used AT MOST ONCE,
        only if `ocr_backend`'s OCR pass also comes back effectively empty.
    """
    # -- Google Docs share link -------------------------------------------- #
    if isinstance(source, str) and _GDOC_RE.match(source.strip()):
        return _from_gdoc(source.strip())

    name = (filename or _name_of(source) or "").lower()
    data = _read_bytes(source)

    if name.endswith(".pdf") or (not name and _looks_like_pdf(data)):
        import fitz  # pymupdf
        with fitz.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
            n_pages = doc.page_count
        if ocr_backend is not None and _is_effectively_empty(text, n_pages):
            ocr_text = _ocr_pdf(data, ocr_backend, fallback=ocr_fallback_backend)
            if ocr_text.strip():
                return ocr_text
        return text

    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    return data.decode("utf-8", errors="ignore")


def _is_effectively_empty(text: str, n_pages: int) -> bool:
    """True if pymupdf's text layer looks like an image-only/scanned PDF."""
    n_pages = max(1, n_pages)
    non_ws = len(re.sub(r"\s+", "", text or ""))
    return (non_ws / n_pages) < OCR_MIN_CHARS_PER_PAGE


# --------------------------------------------------------------------------- #
# OCR fallback (Gemma-multimodal, per-page 5-pass merge; strict single-shot
# Gemini fallback only if Gemma's pass is also empty)
# --------------------------------------------------------------------------- #
def _ocr_pdf(pdf_bytes: bytes, ocr_backend, *, fallback=None) -> str:
    """OCR the first `OCR_MAX_PAGES` pages of *pdf_bytes* with `ocr_backend`.

    Per page, renders 5 crops (full page + 4 halves) at OCR_DPI and issues one
    `generate_multimodal` call per crop, then merges the 5 transcripts with
    line-level dedupe (`_merge_ocr_passes`). This catches text a single
    full-page pass sometimes drops (dense two-column resumes, tiny footer
    text) without ever looping unboundedly — it's always exactly 5 calls/page.

    Never raises: a bad page/call is skipped and OCR continues with whatever
    it could recover (the app's HITL form is where the human fixes the rest).

    If the merged Gemma OCR text still looks empty AND `fallback` is given,
    makes exactly ONE additional `generate_multimodal` call per page (full
    page only, no slicing) — a single last resort, never a retry loop.
    """
    import fitz  # pymupdf

    pages_text: list[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:  # noqa: BLE001 — corrupt/unreadable PDF: nothing to OCR
        return ""

    try:
        n = min(doc.page_count, OCR_MAX_PAGES)
        for i in range(n):
            try:
                page = doc.load_page(i)
                crops = _render_page_crops(page)
            except Exception:  # noqa: BLE001 — bad page: skip, keep going
                continue

            passes: list[str] = []
            for img_bytes in crops:
                try:
                    passes.append(ocr_backend.generate_multimodal(OCR_PROMPT, [img_bytes]) or "")
                except Exception:  # noqa: BLE001 — one bad crop shouldn't kill the page
                    passes.append("")

            page_text = _merge_ocr_passes(passes)

            if not page_text.strip() and fallback is not None:
                # Strictly once per page-that-needs-it, full page only, no retry.
                try:
                    page_text = fallback.generate_multimodal(OCR_PROMPT, [crops[0]]) or ""
                except Exception:  # noqa: BLE001
                    page_text = ""

            if page_text.strip():
                pages_text.append(page_text.strip())
    finally:
        doc.close()

    return "\n\n".join(pages_text)


def _render_page_crops(page) -> list[bytes]:
    """Render *page* (a pymupdf Page) to 5 PNG byte strings: full page, then
    left/right/top/bottom halves, each at OCR_DPI."""
    import fitz  # pymupdf

    rect = page.rect
    boxes = [
        rect,                                                                  # full
        fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width / 2, rect.y1),        # left
        fitz.Rect(rect.x0 + rect.width / 2, rect.y0, rect.x1, rect.y1),        # right
        fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height / 2),       # top
        fitz.Rect(rect.x0, rect.y0 + rect.height / 2, rect.x1, rect.y1),       # bottom
    ]
    out = []
    for box in boxes:
        pix = page.get_pixmap(dpi=OCR_DPI, clip=box)
        out.append(pix.tobytes("png"))
    return out


def _merge_ocr_passes(passes: list[str]) -> str:
    """Merge [full, left, right, top, bottom] OCR transcripts into one text.

    Uses the full-page pass (index 0, if present) as the base line order, then
    appends any lines from the slice passes that aren't already present
    (line-level dedupe, comparing on normalized whitespace/case so trivial
    formatting differences between passes don't produce duplicate lines)."""
    seen: set[str] = set()
    merged: list[str] = []

    def _add_lines(text: str) -> None:
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            key = re.sub(r"\s+", " ", stripped).lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(stripped)

    if passes:
        _add_lines(passes[0])          # full-page pass sets the base order
        for extra in passes[1:]:
            _add_lines(extra)
    return "\n".join(merged)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _name_of(source) -> str:
    name = getattr(source, "name", None)
    if name:
        return str(name)
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    return ""


def _read_bytes(source) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, os.PathLike)) and os.path.exists(os.fspath(source)):
        with open(source, "rb") as fh:
            return fh.read()
    read = getattr(source, "read", None)
    if callable(read):
        data = read()
        return data.encode("utf-8") if isinstance(data, str) else data
    raise TypeError(f"extract_text: unsupported source type {type(source)!r}")


def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _from_gdoc(url: str) -> str:
    """Fetch a link-viewable Google Doc as plain text via its export endpoint.

    Uses the stdlib only (no new dep). Raises with a clear message if the doc
    isn't link-accessible (the caller can fall back to a manual paste)."""
    m = _GDOC_RE.match(url)
    doc_id = m.group(1)
    export = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    import urllib.request
    try:
        with urllib.request.urlopen(export, timeout=30) as resp:  # noqa: S310
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Could not fetch the Google Doc — make sure it is shared as "
            "'Anyone with the link (Viewer)', or paste the text directly."
        ) from exc
    text = raw.decode("utf-8", errors="ignore")
    if "<html" in text[:200].lower():   # got a login/HTML page, not the export
        raise ValueError(
            "The Google Doc is not link-viewable (received a login page). "
            "Share it as 'Anyone with the link', or paste the text directly."
        )
    return text
