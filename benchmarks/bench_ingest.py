"""Per-phase ingest benchmark.

For files, calls POST /api/documents/{doc_id}/reindex and polls
/api/documents every 0.4s to derive per-phase timing from progress_pct
buckets: convert (5–20%), chunk (20–42%), embed (42–96%), save (96–100%).
For URLs, calls POST /api/documents/import-urls and tracks the same
buckets (crawl in place of convert).

Usage:
    poetry run python benchmarks/bench_ingest.py reindex <doc_id> <label>
    poetry run python benchmarks/bench_ingest.py url <url> <label>

Output is a single JSON line so the script composes with shell loops.
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8088"
POLL_S = 0.4


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.load(r)


def _post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"content-type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _doc(doc_id):
    for d in _get("/api/documents"):
        if d["id"] == doc_id:
            return d
    return None


def _bucket_file(pct: int) -> str:
    return (
        "convert" if pct < 20 else
        "chunk"   if pct < 42 else
        "embed"   if pct < 96 else
        "save"
    )


def _bucket_url(pct: int) -> str:
    return (
        "crawl"   if pct < 20 else
        "chunk"   if pct < 42 else
        "embed"   if pct < 96 else
        "save"
    )


def _poll_until_done(doc_id: str, bucket_fn) -> dict:
    t0 = time.perf_counter()
    starts: dict[str, float] = {}
    last_pct = -1
    while True:
        d = _doc(doc_id)
        if d is None:
            return {"status": "missing"}
        pct = int(d.get("progress_pct") or 0)
        if pct != last_pct:
            starts.setdefault(bucket_fn(pct), time.perf_counter() - t0)
            last_pct = pct
        if d["status"] == "ready":
            total = time.perf_counter() - t0
            break
        if d["status"] == "error":
            return {"status": "error", "error": d.get("error_message")}
        time.sleep(POLL_S)

    keys = list(dict.fromkeys(["convert", "crawl", "chunk", "embed", "save"]))
    ordered = [k for k in keys if k in starts]
    durations: dict[str, float | None] = {}
    for i, k in enumerate(ordered):
        next_start = ordered[i + 1] if i + 1 < len(ordered) else None
        durations[k] = round((starts[next_start] if next_start else total) - starts[k], 1)
    return {
        "status": "ready",
        "total_s": round(total, 1),
        "chunks": int(d.get("chunk_count") or 0),
        **durations,
    }


def bench_reindex(doc_id: str, label: str):
    _post(f"/api/documents/{doc_id}/reindex", {})
    return label, _poll_until_done(doc_id, _bucket_file)


def bench_url(url: str, label: str):
    res = _post("/api/documents/import-urls", {"urls": url, "folder": "", "tags": ""})
    doc_id = res["doc_ids"][0]
    out = _poll_until_done(doc_id, _bucket_url)
    out["doc_id"] = doc_id
    return label, out


if __name__ == "__main__":
    kind, *rest = sys.argv[1:]
    if kind == "reindex":
        doc_id, label = rest
        print(json.dumps(bench_reindex(doc_id, label)))
    elif kind == "url":
        url, label = rest
        print(json.dumps(bench_url(url, label)))
    else:
        sys.exit(f"unknown kind: {kind}")
