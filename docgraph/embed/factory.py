from docgraph.config import Config
from docgraph.embed.ollama import OllamaEmbedder
from docgraph.embed.provider import EmbeddingProvider


def create_embedder(cfg: Config) -> EmbeddingProvider:
    if cfg.embed_provider == "ollama":
        return OllamaEmbedder(cfg.ollama_url, cfg.ollama_model)
    if cfg.embed_provider == "openai":
        from docgraph.embed.openai_provider import OpenAIEmbedder
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY required when embed_provider=openai")
        return OpenAIEmbedder(cfg.openai_api_key, cfg.openai_model)
    raise ValueError(f"unknown embed provider: {cfg.embed_provider}")
