import pytest
import httpx
import respx

from docgraph.embed.ollama import OllamaEmbedder


@respx.mock
@pytest.mark.asyncio
async def test_embed_returns_vectors():
    route = respx.post("http://127.0.0.1:11434/api/embed").mock(
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
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={
            "models": [{"name": "nomic-embed-text:latest"}],
        })
    )
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    await embedder.health_check()


@respx.mock
@pytest.mark.asyncio
async def test_embed_batches_large_inputs():
    route = respx.post("http://127.0.0.1:11434/api/embed").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "embeddings": [[0.1]] * len(request.content and 1 or 1),
            },
        )
    )

    def _respond(request: httpx.Request) -> httpx.Response:
        import json

        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1]] * n})

    route.side_effect = _respond
    embedder = OllamaEmbedder("http://127.0.0.1:11434", "nomic-embed-text")
    texts = [f"chunk-{i}" for i in range(30)]
    vecs = await embedder.embed(texts)
    assert len(vecs) == 30
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_health_check_failure():
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(503)
    )
    embedder = OllamaEmbedder("http://localhost:11434", "nomic-embed-text")
    with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
        await embedder.health_check()
