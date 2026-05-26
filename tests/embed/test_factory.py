from boostmcp.config import Config
from boostmcp.embed.factory import create_embedder
from boostmcp.embed.ollama import OllamaEmbedder


def test_factory_creates_ollama_by_default(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    embedder = create_embedder(cfg)
    assert isinstance(embedder, OllamaEmbedder)


def test_factory_rejects_unknown_provider(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir, embed_provider="unknown")
    import pytest
    with pytest.raises(ValueError, match="unknown embed provider"):
        create_embedder(cfg)
