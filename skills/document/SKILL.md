---
name: document
description: Query uploaded documents via DocGraph RAG. Use when the user message starts with /document.
---

# /document — Query DocGraph Knowledge Base

When the user's message starts with `/document`, follow this workflow exactly.

## Trigger

- Message begins with `/document` (with or without space after)
- Example: `/document How do I configure Ollama embedding?`
- Example: `/document --folder DocGraph --tag design What changed in v2?`

## Parse Input

1. Strip the `/document` prefix
2. Extract optional filters:
   - `--folder <name>` → pass as `folder` to MCP
   - `--tag <name>` → pass as `tags: ["<name>"]` to MCP
3. Remaining text is the **query**

## Required MCP Call

Call `search_documents` on the **docgraph** MCP server:

```
search_documents(
  query="<parsed query>",
  folder="<folder or omit>",
  tags=["<tag>"] or omit,
  top_k=5
)
```

## Answer Rules

1. Answer **only** from returned chunks — do not use outside knowledge
2. If `results` is empty or error mentions no matches:
   - Tell user: "No matching documents. Upload files at http://127.0.0.1:8088"
3. If error mentions Ollama:
   - Tell user to start Ollama and run `ollama pull nomic-embed-text`
4. Cite sources inline: `[filename, chunk N]` or `[filename, p.X]` if source_page exists
5. If chunks conflict, note the conflict and cite both sources

## Optional Follow-up

If the user needs full document text, call `get_document(doc_id=...)`.
To see what's indexed, call `list_documents`.
