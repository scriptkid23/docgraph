# Rust Local Embedder (PyO3) Implementation Plan

> **Status:** Implemented

**Goal:** Load ONNX embedding models in-process via Rust (fastembed), replacing Ollama for default local RAG.

**Architecture:** PyO3 crate `docgraph_embed` wraps fastembed `TextEmbedding`. Python `LocalEmbedder` implements `EmbeddingProvider`. Default `DOCGRAPH_EMBED_PROVIDER=local`.

**Build:** `poetry run maturin develop --release -m crates/docgraph-embed/Cargo.toml`

**Models:** Ollama-style names map to ONNX (default `nomic-embed-text` → `NomicEmbedTextV15`, 768-dim).
