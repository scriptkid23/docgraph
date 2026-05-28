from __future__ import annotations

import asyncio
import logging

import httpx

from docgraph.config import normalize_ollama_url

# Batch to avoid huge single /api/embed payloads (large PDFs → many chunks).
EMBED_BATCH_SIZE = 24
_EMBED_TIMEOUT_SEC = 180.0

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    HEALTH_MSG = (
        "Cannot reach Ollama for embeddings. "
        "Start Ollama, run: ollama pull nomic-embed-text, "
        "and set DOCGRAPH_OLLAMA_URL=http://127.0.0.1:11434 (not localhost on Windows)."
    )

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = normalize_ollama_url(base_url.rstrip("/"))
        self._model = model
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(_EMBED_TIMEOUT_SEC)
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _model_installed(self, tags: list[dict]) -> bool:
        want = self._model.split(":")[0].lower()
        for entry in tags:
            name = (entry.get("name") or entry.get("model") or "").lower()
            if name == self._model.lower() or name.split(":")[0] == want:
                return True
        return False

    async def _embed_batch(
        self, client: httpx.AsyncClient, texts: list[str]
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["embeddings"]
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.debug("Ollama embed attempt %d failed: %s", attempt + 1, exc)
        detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
        raise RuntimeError(f"{self.HEALTH_MSG} ({detail})") from last_exc

    async def embed(
        self, texts: list[str], *, for_query: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []
        client = await self._get_client()
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors = await self._embed_batch(client, batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(vectors)} embeddings for {len(batch)} inputs"
                )
            all_vectors.extend(vectors)
        return all_vectors

    async def health_check(self) -> None:
        # Short timeout — uses a one-off client so a hung Ollama doesn't poison
        # the long-lived embedding client.
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{self._base_url}/api/tags")
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"{self.HEALTH_MSG} ({type(exc).__name__}: {exc})"
                ) from exc
            if resp.status_code >= 400:
                raise RuntimeError(self.HEALTH_MSG)
            models = resp.json().get("models") or []
            if not self._model_installed(models):
                raise RuntimeError(
                    f"Embedding model {self._model!r} not found in Ollama. "
                    f"Run: ollama pull {self._model}"
                )
