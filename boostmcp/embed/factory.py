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
