from __future__ import annotations

from typing import Protocol


class RerankerProvider(Protocol):
    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """Return one relevance score per document (higher = more relevant)."""
