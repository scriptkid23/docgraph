"""Cross-config search sweep — compare top-3 chunk ordering across rerank settings.

Re-creates AppState per config (env-driven), warms reranker if enabled, then
runs the same CASES against each. The interesting output is the per-query
top-1 chunk: if rerank changes top-1 chunk within the same doc, it's earning
its latency budget.

Usage:
    poetry run python benchmarks/bench_deep.py
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time

os.environ.setdefault("PYTHONWARNINGS", "ignore")

_records: list[logging.LogRecord] = []


class _Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        _records.append(record)


logging.getLogger("docgraph.mcp.search").setLevel(logging.INFO)
logging.getLogger("docgraph.mcp.search").addHandler(_Capture())

from docgraph.config import load_config  # noqa: E402
from docgraph.web.deps import AppState  # noqa: E402


# Mix of "easy", "specific", and "identifier" queries — varies how RRF clusters
CASES: list[tuple[str, str]] = [
    ("vector embeddings for similarity search",     "ambig: vector embed"),
    ("how to handle callbacks asynchronously",      "ambig: callback async"),
    ("graph traversal algorithm",                   "ambig: graph traverse"),
    ("function scope and binding",                  "ambig: scope/binding"),
    ("Cypher MATCH clause syntax for traversing relationships",
                                                    "specific: KG cypher"),
    ("how do JavaScript closures capture variables", "specific: JS closure"),
    ("node2vec random walk",                        "id: node2vec"),
    ("addEventListener click handler",              "id: addEventListener"),
]

CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("TOP_N=8 GAP=0.5", {
        "DOCGRAPH_RERANK_ENABLED": "true",
        "DOCGRAPH_RERANK_TOP_N": "8",
        "DOCGRAPH_RERANK_SCORE_GAP_RATIO": "0.5",
    }),
    ("TOP_N=4 GAP=0.5", {
        "DOCGRAPH_RERANK_ENABLED": "true",
        "DOCGRAPH_RERANK_TOP_N": "4",
        "DOCGRAPH_RERANK_SCORE_GAP_RATIO": "0.5",
    }),
    ("RERANK=off", {
        "DOCGRAPH_RERANK_ENABLED": "false",
    }),
]


async def _run(label: str, env: dict[str, str]) -> dict:
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = load_config()
        state = AppState.create(cfg)
        if state.reranker is not None and cfg.rerank_enabled:
            await state.reranker.prewarm()
        svc = state.search_service()
        await svc.search("warmup", top_k=1)
        _records.clear()

        rows = []
        for query, qlabel in CASES:
            t0 = time.perf_counter()
            results = await svc.search(query, top_k=5)
            ms = (time.perf_counter() - t0) * 1000
            metric = next(
                (r for r in reversed(_records) if r.getMessage() == "search_metrics"),
                None,
            )
            _records.clear()
            ran = bool(getattr(metric, "rerank_triggered", False)) if metric else False
            top3 = [
                {
                    "doc": r.filename[:30],
                    "chunk_idx": r.chunk_index,
                    "rerank_score": round(r.rerank_score, 3) if r.rerank_score is not None else None,
                    "snippet": r.text[:60].replace("\n", " "),
                }
                for r in results[:3]
            ]
            rows.append({"query": qlabel, "ms": round(ms), "rerank_ran": ran, "top3": top3})
        return {"label": label, "rows": rows, "avg_ms": round(sum(r["ms"] for r in rows) / len(rows))}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def main() -> None:
    runs = []
    for label, env in CONFIGS:
        print(f"\n=== {label} ===")
        d = await _run(label, env)
        runs.append(d)
        print(f"avg={d['avg_ms']}ms")

    # Top-1 chunk diff table
    print("\n\n=== Per-query top-1 chunk (doc#chunk_index) ===")
    header = f"{'query':28}"
    for d in runs:
        header += f" {d['label']:<20}"
    print(header)
    print("-" * len(header))
    for i, (_, qlabel) in enumerate(CASES):
        row = f"{qlabel:28}"
        cells = []
        for d in runs:
            t = d["rows"][i]["top3"][0]
            cells.append(f"{t['doc'][:14]}#{t['chunk_idx']}")
        for c in cells:
            row += f" {c:<20}"
        same = " same" if len(set(cells)) == 1 else ""
        print(row + same)

    print("\n=== Latency comparison ===")
    for d in runs:
        per = [r["ms"] for r in d["rows"]]
        print(f"{d['label']:20} avg={d['avg_ms']:>5}ms  min={min(per):>5} max={max(per):>5}")

    out_path = "/tmp/bench_deep_results.json"
    with open(out_path, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
