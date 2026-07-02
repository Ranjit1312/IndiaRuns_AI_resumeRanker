"""Fetch complete, verbatim JD text for each posting in _manifest.json.

Uses each site's real source (Greenhouse API, Workday CXS API, or embedded
schema.org JobPosting JSON-LD) rather than a summarizer, so we capture the full
JD. Writes data/eval_jds/<slug>/jd.md + source.json. Prints a per-job status
table; full HTML never touches the caller's context.

    python tools/fetch_jd.py            # all jobs in the manifest
    python tools/fetch_jd.py <slug>...  # only these slugs
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "data", "eval_jds")
MANIFEST = os.path.join(EVAL, "_manifest.json")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _get(url: str, headers: dict | None = None, data: bytes | None = None) -> str:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA,
                                 "Accept": "*/*", **(headers or {})})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", errors="ignore")


def html_to_text(h: str) -> str:
    # Greenhouse returns HTML-ESCAPED content (&lt;p&gt;…); reveal real tags first
    # so the tag-stripping below can remove them. Harmless for already-raw HTML.
    if "&lt;" in h:
        h = html.unescape(h)
    h = re.sub(r"(?i)<\s*br\s*/?>", "\n", h)
    h = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", h)
    h = re.sub(r"(?i)</\s*(p|div|h[1-6]|ul|ol|section)\s*>", "\n", h)
    h = re.sub(r"(?i)<\s*(h[1-6])[^>]*>", "\n\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = html.unescape(h)
    h = re.sub(r"[ \t]+\n", "\n", h)
    h = re.sub(r"\n{3,}", "\n\n", h)
    return h.strip()


def _jsonld_description(page: str) -> str | None:
    for m in re.finditer(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
                         page, re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        for cand in (obj if isinstance(obj, list) else [obj]):
            if isinstance(cand, dict) and cand.get("description"):
                title = cand.get("title", "")
                loc = ""
                jl = cand.get("jobLocation")
                if isinstance(jl, dict):
                    loc = str(jl.get("address", {}).get("addressLocality", ""))
                body = html_to_text(cand["description"])
                head = f"# {title}\n" if title else ""
                head += f"_Location: {loc}_\n\n" if loc else ""
                return head + body
    return None


# ---- per-source strategies -------------------------------------------------
def from_greenhouse(token: str, job_id: str, page_url: str | None = None) -> str:
    api = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{job_id}?content=true"
    try:
        obj = json.loads(_get(api))
        title = obj.get("title", "")
        loc = (obj.get("location") or {}).get("name", "")
        body = html_to_text(obj.get("content", ""))
        return f"# {title}\n_Location: {loc}_\n\n{body}"
    except urllib.error.HTTPError:
        # API 404 (closed/moved id): fall back to the rendered board page's JSON-LD
        if page_url:
            desc = _jsonld_description(_get(page_url))
            if desc and len(desc) > 400:
                return desc
        raise


def from_workday(url: str) -> str:
    # host/<lang>/<site>/job/<path>  ->  host/wday/cxs/<tenant>/<site>/job/<path>
    m = re.match(r"https?://([^/]+)/([^/]+)/([^/]+)/job/(.+)$", url)
    host, _lang, site, path = m.group(1), m.group(2), m.group(3), m.group(4)
    tenant = host.split(".")[0]
    api = f"https://{host}/wday/cxs/{tenant}/{site}/job/{path}"
    obj = json.loads(_get(api, headers={"Accept": "application/json"}))
    info = obj.get("jobPostingInfo", obj)
    title = info.get("title", "")
    loc = info.get("location", "")
    body = html_to_text(info.get("jobDescription", ""))
    return f"# {title}\n_Location: {loc}_\n\n{body}"


def from_amazon(url: str) -> str:
    jid = re.search(r"/jobs/(\d+)", url).group(1)
    obj = json.loads(_get(f"https://www.amazon.jobs/en/jobs/{jid}.json"))
    title = obj.get("title", "")
    loc = obj.get("normalized_location") or obj.get("location", "")
    parts = [f"# {title}\n_Location: {loc}_\n"]
    for label, key in [("Description", "description"),
                       ("Basic Qualifications", "basic_qualifications"),
                       ("Preferred Qualifications", "preferred_qualifications")]:
        if obj.get(key):
            parts.append(f"\n## {label}\n{html_to_text(obj[key])}")
    return "\n".join(parts)


def from_google(url: str) -> str:
    jid = re.search(r"results/(\d+)", url).group(1)
    # careers.google.com v3 job API returns the full description HTML
    obj = json.loads(_get(f"https://careers.google.com/api/v3/jobs/{jid}/",
                          headers={"Accept": "application/json"}))
    j = obj.get("job", obj)
    title = j.get("title", "")
    body = html_to_text(j.get("description", ""))
    quals = html_to_text(j.get("qualifications", "")) if j.get("qualifications") else ""
    resp = html_to_text(j.get("responsibilities", "")) if j.get("responsibilities") else ""
    out = [f"# {title}\n"]
    if quals:
        out.append(f"\n## Qualifications\n{quals}")
    if resp:
        out.append(f"\n## Responsibilities\n{resp}")
    out.append(f"\n## Description\n{body}")
    return "\n".join(out)


def fetch_one(entry: dict) -> tuple[str, str]:
    """Return (markdown, method). Raises on total failure."""
    url = entry["url"]
    m = re.search(r"greenhouse\.io/([^/]+)/jobs/(\d+)", url)
    if m:
        return from_greenhouse(m.group(1), m.group(2), url), "greenhouse"
    if "databricks.com/company/careers" in url:
        jid = re.search(r"(\d{6,})", url).group(1)
        return from_greenhouse("databricks", jid, url), "greenhouse(databricks)"
    if "myworkdayjobs.com" in url:
        return from_workday(url), "workday-cxs"
    if "amazon.jobs" in url:
        return from_amazon(url), "amazon-json"
    if "google.com" in url and re.search(r"results/(\d+)", url):
        try:
            return from_google(url), "google-v3"
        except Exception:  # noqa: BLE001 — fall through to generic
            pass
    # generic: fetch page, prefer JSON-LD JobPosting
    page = _get(url)
    desc = _jsonld_description(page)
    if desc and len(desc) > 400:
        return desc, "json-ld"
    raise RuntimeError(f"no extractor matched (page {len(page)} bytes)")


def main(argv):
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    want = set(argv) if argv else None
    print(f"{'slug':45} {'method':22} chars status")
    for e in manifest:
        if want and e["slug"] not in want:
            continue
        d = os.path.join(EVAL, e["slug"])
        os.makedirs(d, exist_ok=True)
        try:
            md, method = fetch_one(e)
            md = f"# {e['role']} — {e['org']}\nSource: {e['url']}\n\n" + md
            open(os.path.join(d, "jd.md"), "w", encoding="utf-8").write(md)
            json.dump({"org": e["org"], "role_title": e["role"], "url": e["url"],
                       "fetched_at": "2026-07-02", "fetch_method": method},
                      open(os.path.join(d, "source.json"), "w", encoding="utf-8"), indent=2)
            print(f"{e['slug']:45} {method:22} {len(md):5}  OK")
        except Exception as exc:  # noqa: BLE001
            print(f"{e['slug']:45} {'-':22} {'-':>5}  FAIL: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main(sys.argv[1:])
