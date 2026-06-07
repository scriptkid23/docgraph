"""Search benchmark — latency + reranker gate decisions.

Loads AppState in-process so it can read the structured `search_metrics`
log records (rerank_triggered, rerank_reason, branch sizes) the running
SearchService emits at INFO level. The data dir is read from the active
config (~/.docgraph by default), so this hits whatever corpus the server
is using.

Usage:
    poetry run python benchmarks/bench_search.py
"""
from __future__ import annotations
import asyncio
import logging
import os
import time

os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Capture structured metric records from the search logger
_records: list[logging.LogRecord] = []


class _Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _records.append(record)


logging.getLogger("docgraph.mcp.search").setLevel(logging.INFO)
logging.getLogger("docgraph.mcp.search").addHandler(_Capture())

from docgraph.config import load_config  # noqa: E402
from docgraph.web.deps import AppState  # noqa: E402


# Edit to match what your corpus actually contains.
CASES: list[tuple[str, str, str]] = [
    ("what is a knowledge graph",            "Knowledge_Graphs",            "KG: semantic"),
    ("graph embeddings retrieval",           "Knowledge_Graphs",            "KG: tech phrase"),
    ("node2vec",                             "Knowledge_Graphs",            "KG: identifier"),
    ("Cypher MATCH clause",                  "Knowledge_Graphs",            "KG: mix"),
    ("how do closures work in JavaScript",   "Head_First_JavaScript",       "JS: semantic"),
    ("async await example",                  "Head_First_JavaScript",       "JS: phrase"),
    ("Array.prototype.map",                  "Head_First_JavaScript",       "JS: identifier"),
    ("event listener attach DOM",            "Head_First_JavaScript",       "JS: technical"),
]


async def main() -> None:
    cfg = load_config()
    state = AppState.create(cfg)
    print(
        f"cfg.rerank_enabled={cfg.rerank_enabled} "
        f"cfg.rerank_top_n={cfg.rerank_top_n} "
        f"cfg.rerank_score_gap_ratio={cfg.rerank_score_gap_ratio}"
    )
    if state.reranker is not None and cfg.rerank_enabled:
        await state.reranker.prewarm()
    svc = state.search_service()

    # Warmup so model load doesn't skew the first row
    await svc.search("warmup", top_k=1)
    _records.clear()

    print(
        f"\n{'label':22} {'ms':>6} {'ran':>5} {'reason':>22} "
        f"{'match':>5} | top filename"
    )
    print("-" * 110)

    latencies: list[float] = []
    correct = 0
    for query, expect_substr, label in CASES:
        t0 = time.perf_counter()
        results = await svc.search(query, top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        metric = next(
            (r for r in reversed(_records) if r.getMessage() == "search_metrics"),
            None,
        )
        _records.clear()
        ran = bool(getattr(metric, "rerank_triggered", False)) if metric else False
        reason = getattr(metric, "rerank_reason", "") if metric else ""

        top = results[0] if results else None
        if top is None:
            print(f"{label:22} {ms:>6.0f}     - {'':>22}     - | no results")
            continue
        ok = expect_substr in top.filename
        correct += int(ok)
        mark = "OK" if ok else "no"
        print(
            f"{label:22} {ms:>6.0f} {str(ran):>5} {reason:>22} "
            f"{mark:>5} | {top.filename[:42]}"
        )

    avg = sum(latencies) / len(latencies)
    print(f"\nSummary: {correct}/{len(CASES)} correct doc, avg {avg:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
