# DocGraph benchmarks

Reusable scripts that drove the latency tuning in `feat/hybrid-search-rerank`.
They hit a running DocGraph server (default `http://127.0.0.1:8088`) or load
`AppState` directly to bypass HTTP.

## Scripts

| Script | What it measures | How |
|---|---|---|
| `bench_ingest.py` | Per-phase ingest timing (convert / chunk / embed / save) for files and URLs | Calls `/reindex` or `/import-urls`, polls `/api/documents` every 0.4s |
| `bench_search.py` | End-to-end search latency + whether reranker fired and why | Loads `AppState` directly, logs `search_metrics` records |
| `bench_deep.py` | Cross-config sweep — compares top-3 chunk ordering across rerank settings | Runs the same queries against three configurations and diffs |

All scripts are stand-alone (`poetry run python benchmarks/<script>.py`) and
assume the server has READY docs to query/reindex against.

## Reference numbers — 2026-06-07

5 docs, 1530 chunks, MacBook Pro M-series, local Rust ONNX embedder (768-dim
`nomic-embed-text`), CPU `bge-reranker-v2-m3`.

### Ingest

| Source | Size | Chunks | Total | Convert/Crawl | Embed | s/chunk |
|---|---:|---:|---:|---:|---:|---:|
| Blockchain_Trilemma.pdf | 70 KB | 2 | 1.2s | 0.4s | 0.8s | 0.40 |
| Optimistic_Rollup.pdf | 374 KB | 12 | 5.8s | 0.9s | 4.9s | 0.41 |
| Knowledge_Graphs.pdf | 27 MB | 751 | 7m 01s | 67s | 354s | 0.47 |
| Head_First_JS.pdf | 97 MB | 620 | 6m 42s | 124s | 278s | 0.45 |
| docs.python.org/asyncio-task | – | 113 | 48s | 2s | 46s | 0.41 |
| wikipedia/Reinforcement_learning | – | 168 | 86s | 3s | 83s | 0.50 |
| docs.docker.com/install | – | 57 | 28s | 3s | 26s | 0.45 |

**Embedding dominates (70–84% of total).** Throughput on CPU: ~0.42 s/chunk =
~144 chunks/min. Convert phase non-linear in MB; chunk + save are negligible.

### Search

8-query suite, top_k=5, after `_to_result` fix.

| Config | Avg latency | Doc-level | Chunk-level (vs TOP_N=8 baseline) |
|---|---:|:---:|:---:|
| TOP_N=8 GAP=0.5 (pre-tuning) | 7513 ms | 8/8 | baseline |
| **TOP_N=4 GAP=0.5** (new default) | **3957 ms** | **8/8** | **7/8 identical chunk** |
| RERANK=off | 26 ms | 8/8 | 3/8 identical chunk |

`TOP_N=4` is a strict win: same doc, virtually same chunk, half the latency.
Disabling rerank loses chunk-level precision for ~5/8 queries (top doc still
correct) — acceptable for browsing, not for "quote the best paragraph" flows.

`GAP_RATIO` had no observable effect on this corpus (RRF top-1 never dominated
by >50%, so rerank fired on every query under all tested ratios). The knob
may still help on larger corpora where some queries have a clearly winning
candidate after fusion.
