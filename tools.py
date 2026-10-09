"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree  # (arXiv answers with Atom XML)

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None:
                delay = min(float(e.retry_after), cap)
            else:
                delay = min(base * 2**attempt, cap)
                delay = min(delay + random.uniform(0, delay), cap)
            time.sleep(max(delay, 0.0))


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_after(response):
    """Seconds from a numeric Retry-After header, or None."""
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _request(method, url, **kwargs):
    """One HTTP call that turns transient failures into RetryableError; wrap it in with_retry."""
    kwargs.setdefault("timeout", 30.0)
    try:
        response = httpx.request(method, url, **kwargs)
    except httpx.TransportError as e:
        raise RetryableError(f"{type(e).__name__}: {e}") from e
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url}", retry_after=_retry_after(response))
    response.raise_for_status()
    return response


# ---- TODO 2: arXiv ----
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_OPERATORS = {"AND", "OR", "ANDNOT", "NOT"}
ARXIV_MIN_GAP = 3.0  # seconds between two arXiv calls (arXiv API etiquette)
_arxiv_last_call = 0.0


def _arxiv_get(params):
    """One arXiv request, spaced at least ARXIV_MIN_GAP seconds after the previous one."""
    global _arxiv_last_call
    wait = _arxiv_last_call + ARXIV_MIN_GAP - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        return _request("GET", ARXIV_URL, params=params)
    finally:
        _arxiv_last_call = time.monotonic()


def _clean(text):
    """Collapse newlines and runs of whitespace into single spaces."""
    return " ".join((text or "").split())


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    try:
        terms = [t for t in re.findall(r"[\w-]+", query) if t.strip("-") and t.upper() not in ARXIV_OPERATORS]
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{t.strip('-')}" for t in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max(1, min(int(max_results), 30)),
        }
        response = with_retry(lambda: _arxiv_get(params), attempts=6, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.content)
        records = []
        for entry in root.findall(f"{ATOM}entry"):
            raw_id = _clean(entry.findtext(f"{ATOM}id"))
            if not raw_id:
                continue
            arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])
            records.append({
                "id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "published": _clean(entry.findtext(f"{ATOM}published"))[:10],
                "title": _clean(entry.findtext(f"{ATOM}title")),
                "summary": _clean(entry.findtext(f"{ATOM}summary"))[:600],
            })
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 3: Hugging Face ----
def _hf_records(items):
    """Map Hugging Face paper items to compact records; items without paper.id are skipped."""
    records = []
    for item in items or []:
        paper = (item or {}).get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        records.append({
            "id": paper_id,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clean(paper.get("title") or item.get("title")),
            "summary": _clean(paper.get("ai_summary") or paper.get("summary") or item.get("summary"))[:600],
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0,
        })
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(int(limit), 100))}
        if date.strip():
            params["date"] = date.strip()
        items = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params)).json()
        records = _hf_records(items)
        if keyword.strip():
            needle = keyword.strip().lower()
            records = [r for r in records if needle in f"{r['title']} {r['summary']}".lower()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not query.strip():
            return "NO RESULTS"
        n = max(1, min(int(limit), 50))
        items = with_retry(lambda: _request("GET", HF_SEARCH_URL, params={"q": query.strip(), "limit": n})).json()
        records = _hf_records(items)[:n]
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
WEB_FETCH_MAX_CHARS = 12000
EXA_RATE_LIMIT_TEXT = re.compile(r"rate[ _-]?limit", re.IGNORECASE)


def _redact(text):
    """Hide EXA_API_KEY (and any exaApiKey=... URL parameter) from text that goes back to the agent."""
    key = os.environ.get("EXA_API_KEY", "").strip()
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", text)


def _exa_url():
    key = os.environ.get("EXA_API_KEY", "").strip()
    return f"{EXA_URL}?exaApiKey={key}" if key else EXA_URL


def _parse_mcp_body(response):
    """MCP answers with server-sent events (`data: {...}` lines) or, for some errors, plain JSON."""
    body = response.text.strip()
    if body.startswith("{"):
        return json.loads(body)
    messages = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:") and line[5:].strip()]
    if not messages:
        raise ValueError(f"no data in MCP response (HTTP {response.status_code})")
    return messages[-1]


def _is_rate_limited(result, text):
    """The free tier can answer HTTP 200 with a rate-limit notice as the 'content' and a flag in result._meta."""
    meta = result.get("_meta") or {}
    for key, value in meta.items():
        if value and (EXA_RATE_LIMIT_TEXT.search(str(key)) or EXA_RATE_LIMIT_TEXT.search(json.dumps(value))):
            return True
    return bool(result.get("isError")) and bool(EXA_RATE_LIMIT_TEXT.search(text))


def _exa_once(name, arguments):
    """One JSON-RPC tools/call to Exa; rate limits and transient failures raise RetryableError."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    try:
        response = _request("POST", _exa_url(), json=payload, headers=headers, timeout=60.0)
    except RetryableError as e:
        # Exa sends Retry-After: 1 while its limit lasts ~20 s or more: use the exponential backoff instead.
        raise RetryableError(_redact(str(e))) from None
    message = _parse_mcp_body(response)
    if message.get("error"):
        error_text = str(message["error"].get("message") or message["error"])
        if EXA_RATE_LIMIT_TEXT.search(error_text):
            raise RetryableError("Exa rate limit")
        raise RuntimeError(f"Exa MCP error: {error_text[:300]}")
    result = message.get("result") or {}
    text = "\n\n".join(c.get("text", "") for c in result.get("content") or [] if c.get("type") == "text").strip()
    if _is_rate_limited(result, text):
        raise RetryableError("Exa rate limit")
    if result.get("isError"):
        raise RuntimeError(f"Exa tool error: {text[:300]}")
    return text


def _exa_call(name, arguments):
    return with_retry(lambda: _exa_once(name, arguments), attempts=8, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        if not query.strip():
            return "NO RESULTS"
        arguments = {
            "query": query.strip(),
            "objective": objective.strip() or f"Find the most relevant, authoritative web pages about: {query.strip()}",
            "numResults": max(1, min(int(num_results), 10)),
        }
        text = _exa_call("web_search_exa", arguments)
        return text or "NO RESULTS"
    except Exception as e:
        return _redact(f"ERROR: {type(e).__name__}: {e}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    try:
        if not url.strip():
            return "NO RESULTS"
        text = _exa_call("web_fetch_exa", {"urls": [url.strip()]})
        if not text:
            return "NO RESULTS"
        if len(text) > WEB_FETCH_MAX_CHARS:
            text = text[:WEB_FETCH_MAX_CHARS] + "\n\n[... truncated ...]"
        return text
    except Exception as e:
        return _redact(f"ERROR: {type(e).__name__}: {e}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
