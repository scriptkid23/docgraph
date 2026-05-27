# URL Import (crawl4ai) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Import a textarea of URLs, crawl each page with crawl4ai to markdown, then chunk and embed like file uploads.

**Architecture:** New `UrlCrawler` wraps crawl4ai; `Indexer.index_urls_batch` reuses one browser session; `POST /api/documents/import-urls` queues background jobs; SQLite stores `source_type` + `source_url` for re-crawl on reindex.

**Tech Stack:** crawl4ai (optional extra), Playwright/Chromium, existing chunk/embed pipeline

---

## Implemented

- [x] Schema: `source_type`, `source_url` on documents
- [x] `docgraph/ingest/urls.py` — parse, validate, SSRF block
- [x] `docgraph/ingest/crawler.py` — crawl4ai wrapper
- [x] `Indexer.index_markdown`, `index_url`, `index_urls_batch`
- [x] `POST /api/documents/import-urls`
- [x] Frontend `LinkImportSection`
- [x] Tests + README
