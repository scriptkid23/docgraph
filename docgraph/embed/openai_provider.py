from __future__ import annotations

from openai import AsyncOpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(
        self, texts: list[str], *, for_query: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    async def health_check(self) -> None:
        await self.embed(["health"])
