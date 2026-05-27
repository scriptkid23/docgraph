import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from docgraph.embed.openai_provider import OpenAIEmbedder


@pytest.mark.asyncio
async def test_openai_embed():
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]
    with patch("docgraph.embed.openai_provider.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.embeddings.create = AsyncMock(return_value=mock_response)
        embedder = OpenAIEmbedder("sk-test", "text-embedding-3-small")
        vecs = await embedder.embed(["a", "b"])
        assert len(vecs) == 2
