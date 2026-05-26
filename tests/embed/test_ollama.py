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
    with pytest.raises(RuntimeError, match="Start Ollama"):
        await embedder.health_check()
