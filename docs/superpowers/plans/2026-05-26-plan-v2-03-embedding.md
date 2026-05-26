# Plan v2-03 — Embedding Providers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pluggable embedding providers — Ollama (default) and OpenAI (optional).

**Architecture:** `EmbeddingProvider` Protocol in `boostmcp/embed/provider.py`. Factory selects provider from config. Ollama calls `POST /api/embed`. OpenAI uses optional extra dependency.

**Tech Stack:** httpx, openai (optional extra)

**Depends on:** Plan v2-01  
**Blocks:** Plan v2-04

**Spec refs:** §7 Embedding Provider, §8.2 Error Behavior

---

## File Structure

```
boostmcp/embed/
├── __init__.py
├── provider.py
├── ollama.py
├── openai_provider.py
└── factory.py
tests/embed/
├── test_ollama.py
└── test_factory.py
```

Add dependencies: `httpx = "^0.28"`. Optional extra `openai` in pyproject.toml.

---

### Task 1: Embedding protocol and factory

**Files:**
- Create: `boostmcp/embed/provider.py`
- Create: `boostmcp/embed/factory.py`
- Create: `tests/embed/test_factory.py`

- [ ] **Step 1: Write failing test**

```python
# tests/embed/test_factory.py
import pytest

from boostmcp.config import Config
from boostmcp.embed.factory import create_embedder
from boostmcp.embed.ollama import OllamaEmbedder


def test_factory_creates_ollama_by_default(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    embedder = create_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)


def test_factory_rejects_unknown_provider(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir, embed_provider="unknown")
    with pytest.raises(ValueError, match="unknown embed provider"):
        create_embedder(cfg)
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
# boostmcp/embed/provider.py
from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def health_check(self) -> None: ...
```

```python
# boostmcp/embed/factory.py
from boostmcp.config import Config
from boostmcp.embed.ollama import OllamaEmbedder
from boostmcp.embed.provider import EmbeddingProvider


def create_embedder(cfg: Config) -> EmbeddingProvider:
    if cfg.embed_provider == "ollama":
        return OllamaEmbedder(cfg.ollama_url, cfg.ollama_model)
    if cfg.embed_provider == "openai":
        from boostmcp.embed.openai_provider import OpenAIEmbedder
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when embed_provider=openai")
        return OpenAIEmbedder(cfg.openai_api_key, cfg.openai_model)
    raise ValueError(f"unknown embed provider: {cfg.embed_provider}")
```

- [ ] **Step 4: Create stub OllamaEmbedder** (minimal for factory test)

```python
# boostmcp/embed/ollama.py
from __future__ import annotations

import httpx


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def health_check(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 5: Run factory tests — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add boostmcp/embed/ tests/embed/test_factory.py
git commit -m "feat: add embedding provider protocol and factory"
```

---

### Task 2: Ollama embedder

**Files:**
- Modify: `boostmcp/embed/ollama.py`
- Create: `tests/embed/test_ollama.py`

- [ ] **Step 1: Write failing test with httpx mock**

```python
# tests/embed/test_ollama.py
import pytest
import httpx
import respx

from boostmcp.embed.ollama import OllamaEmbedder


@respx.mock
@pytest.mark.asyncio
async def test_embed_returns_vectors():
    route = respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(200, json={
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        })
    )
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    vecs = await embedder.embed(["hello", "world"])
    assert len(vecs) == 2
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_health_check_success():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    await embedder.health_check()


@respx.mock
@pytest.mark.asyncio
async def test_health_check_failure():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(503)
    )
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    with pytest.raises(RuntimeError, match="Ollama unavailable"):
        await embedder.health_check()
```

Add dev dependency: `poetry add --group dev respx`

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
# boostmcp/embed/ollama.py  (replace stub)
from __future__ import annotations

import httpx


class OllamaEmbedder:
    HEALTH_MSG = (
        "Start Ollama and run: ollama pull nomic-embed-text"
    )

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{self._base_url}/api/embed",
                        json={"model": self._model, "input": texts},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data["embeddings"]
                except httpx.HTTPError:
                    if attempt == 2:
                        raise RuntimeError(self.HEALTH_MSG) from None
        return []

    async def health_check(self) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self._base_url}/api/tags")
            if resp.status_code >= 400:
                raise RuntimeError(self.HEALTH_MSG)
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add boostmcp/embed/ollama.py tests/embed/test_ollama.py pyproject.toml poetry.lock
git commit -m "feat: implement Ollama embedding provider with retry"
```

---

### Task 3: OpenAI embedder (optional)

**Files:**
- Create: `boostmcp/embed/openai_provider.py`
- Create: `tests/embed/test_openai.py`

- [ ] **Step 1: Add optional extra to pyproject.toml**

```toml
[tool.poetry.extras]
openai = ["openai"]

[tool.poetry.dependencies]
openai = { version = "^1.0", optional = true }
```

- [ ] **Step 2: Write failing test**

```python
# tests/embed/test_openai.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from boostmcp.embed.openai_provider import OpenAIEmbedder


@pytest.mark.asyncio
async def test_openai_embed():
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]
    with patch("boostmcp.embed.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.embeddings.create = AsyncMock(return_value=mock_response)
        embedder = OpenAIEmbedder("sk-test", "text-embedding-3-small")
        vecs = await embedder.embed(["a", "b"])
        assert len(vecs) == 2
```

- [ ] **Step 3: Implement**

```python
# boostmcp/embed/openai_provider.py
from __future__ import annotations

from openai import AsyncOpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    async def health_check(self) -> None:
        await self.embed(["health"])
```

- [ ] **Step 4: Run tests**

```bash
poetry install -E openai
poetry run pytest tests/embed/ -v
```

- [ ] **Step 5: Commit**

```bash
git add boostmcp/embed/openai_provider.py tests/embed/test_openai.py pyproject.toml
git commit -m "feat: add optional OpenAI embedding provider"
```
