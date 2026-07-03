"""Offline tests for the OCR fallback path in harness/ingest.py.

No network: builds a real (but text-layer-free) PDF with pymupdf so
`extract_text`'s pymupdf pass genuinely returns ~nothing, then feeds a
`MockBackend` configured with a `multimodal_responder` that returns known,
scripted text per call — proving the 5-pass merge/dedupe logic and the
strictly-once Gemini fallback, without ever calling a real OCR model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz

from harness.backends import MockBackend
from harness.ingest import (OCR_MAX_PAGES, _is_effectively_empty,
                            _merge_ocr_passes, _ocr_pdf, extract_text)


def _blank_pdf_bytes(n_pages: int = 1) -> bytes:
    """A valid PDF with pages but NO text layer (pymupdf get_text() -> "")."""
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page()
        # draw a rectangle so the page isn't literally content-free, but no text
        page.draw_rect(fitz.Rect(10, 10, 200, 200))
    data = doc.tobytes()
    doc.close()
    return data


def _text_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a normal text-layer PDF with plenty of content.")
    data = doc.tobytes()
    doc.close()
    return data


# --------------------------------------------------------------------------- #
# _is_effectively_empty / detection
# --------------------------------------------------------------------------- #
def test_effectively_empty_detection():
    assert _is_effectively_empty("", 1) is True
    assert _is_effectively_empty("   \n  ", 3) is True
    assert _is_effectively_empty("x" * 19, 1) is True    # just under threshold
    assert _is_effectively_empty("x" * 21, 1) is False   # just over threshold
    assert _is_effectively_empty("word " * 100, 5) is False


# --------------------------------------------------------------------------- #
# merge/dedupe
# --------------------------------------------------------------------------- #
def test_merge_ocr_passes_dedupes_case_and_whitespace():
    full = "Line One\nLine Two\n"
    left = "line one\nLine Three"      # dup of "Line One" (case-insensitive)
    right = "Line   Two\nLine Four"    # dup of "Line Two" (whitespace-normalized)
    top = "Line Five"
    bottom = ""
    merged = _merge_ocr_passes([full, left, right, top, bottom])
    lines = merged.splitlines()
    assert lines == ["Line One", "Line Two", "Line Three", "Line Four", "Line Five"]


def test_merge_ocr_passes_empty_list():
    assert _merge_ocr_passes([]) == ""
    assert _merge_ocr_passes(["", "", "", "", ""]) == ""


# --------------------------------------------------------------------------- #
# extract_text: text-layer PDF is untouched by OCR (no ocr_backend calls)
# --------------------------------------------------------------------------- #
def test_text_layer_pdf_never_triggers_ocr():
    calls = []

    def multimodal_responder(prompt, images, system):
        calls.append(1)
        return "SHOULD NOT BE CALLED"

    backend = MockBackend(lambda p, s: "{}", multimodal_responder=multimodal_responder)
    text = extract_text(_text_pdf_bytes(), filename="resume.pdf", ocr_backend=backend)
    assert "normal text-layer PDF" in text
    assert calls == [], "OCR must not run when pymupdf already found real text"


# --------------------------------------------------------------------------- #
# extract_text: scanned PDF triggers OCR, 5 calls/page, merged result returned
# --------------------------------------------------------------------------- #
def test_scanned_pdf_triggers_ocr_5_calls_per_page_and_merges():
    call_log = []

    def multimodal_responder(prompt, images, system):
        call_log.append(prompt)
        idx = len(call_log) - 1   # 0=full,1=left,2=right,3=top,4=bottom (page 1)
        canned = {
            0: "Name: Jane Doe\nExperience: 5 years",
            1: "Name: Jane Doe",                 # dup of a full-page line
            2: "Experience: 5 years",             # dup
            3: "Skills: Python",                  # new
            4: "",                                # empty crop
        }
        return canned[idx]

    backend = MockBackend(lambda p, s: "{}", multimodal_responder=multimodal_responder)
    pdf_bytes = _blank_pdf_bytes(n_pages=1)

    text = extract_text(pdf_bytes, filename="scanned.pdf", ocr_backend=backend)

    assert len(call_log) == 5, "one generate_multimodal call per crop (5 per page)"
    assert "Name: Jane Doe" in text
    assert "Experience: 5 years" in text
    assert "Skills: Python" in text
    # dedupe worked: "Name: Jane Doe" should appear exactly once
    assert text.count("Name: Jane Doe") == 1


def test_ocr_caps_at_max_pages():
    call_log = []

    def multimodal_responder(prompt, images, system):
        call_log.append(1)
        return "some text"

    backend = MockBackend(lambda p, s: "{}", multimodal_responder=multimodal_responder)
    pdf_bytes = _blank_pdf_bytes(n_pages=OCR_MAX_PAGES + 3)
    _ocr_pdf(pdf_bytes, backend)
    assert len(call_log) == OCR_MAX_PAGES * 5


# --------------------------------------------------------------------------- #
# Gemini fallback: strictly once, only when Gemma's merged pass is empty
# --------------------------------------------------------------------------- #
def test_fallback_not_invoked_when_gemma_ocr_succeeds():
    def gemma_responder(prompt, images, system):
        return "Gemma got the text just fine"

    fallback_calls = []
    def fallback_responder(prompt, images, system):
        fallback_calls.append(1)
        return "SHOULD NOT BE CALLED"

    gemma = MockBackend(lambda p, s: "{}", name="gemma", multimodal_responder=gemma_responder)
    gemini = MockBackend(lambda p, s: "{}", name="gemini", multimodal_responder=fallback_responder)

    pdf_bytes = _blank_pdf_bytes(n_pages=1)
    text = _ocr_pdf(pdf_bytes, gemma, fallback=gemini)
    assert "Gemma got the text" in text
    assert fallback_calls == [], "fallback must not fire when Gemma OCR already succeeded"


def test_fallback_invoked_exactly_once_per_page_when_gemma_empty():
    def gemma_responder(prompt, images, system):
        return ""   # Gemma OCR fails on every crop

    fallback_calls = []
    def fallback_responder(prompt, images, system):
        fallback_calls.append(len(images))
        return "Gemini rescued this page"

    gemma = MockBackend(lambda p, s: "{}", name="gemma", multimodal_responder=gemma_responder)
    gemini = MockBackend(lambda p, s: "{}", name="gemini", multimodal_responder=fallback_responder)

    pdf_bytes = _blank_pdf_bytes(n_pages=2)
    text = _ocr_pdf(pdf_bytes, gemma, fallback=gemini)

    assert len(fallback_calls) == 2, "exactly one fallback call per page needing it, no loop"
    assert all(n == 1 for n in fallback_calls), "fallback call uses a single full-page image only"
    assert text.count("Gemini rescued this page") == 2


def test_ocr_never_raises_on_bad_page(monkeypatch):
    """A crop/call that raises must not kill the whole OCR pass."""
    def flaky_responder(prompt, images, system):
        raise RuntimeError("simulated model hiccup")

    backend = MockBackend(lambda p, s: "{}", multimodal_responder=flaky_responder)
    pdf_bytes = _blank_pdf_bytes(n_pages=1)
    # Should not raise, just return "" (nothing recovered).
    text = _ocr_pdf(pdf_bytes, backend)
    assert text == ""


def test_ocr_skips_gracefully_on_corrupt_pdf():
    backend = MockBackend(lambda p, s: "{}",
                          multimodal_responder=lambda p, i, s: "x")
    text = _ocr_pdf(b"not a real pdf", backend)
    assert text == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except TypeError:
                continue
            print(f"PASS {name}")
    print("all ok")
