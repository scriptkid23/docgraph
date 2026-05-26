from __future__ import annotations

import httpx


class OllamaEmbedder:
    HEALTH_MSG = "Start Ollama and run: ollama pull nomic-embed-text"

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
