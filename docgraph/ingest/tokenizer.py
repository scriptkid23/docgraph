from __future__ import annotations

import logging
import math
from typing import Protocol

from docgraph.config import Config

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return a prefix of text containing at most max_tokens."""

    def truncate_tail(self, text: str, max_tokens: int) -> str:
        """Return a suffix of text containing at most max_tokens."""


class CharRatioCounter:
    """Fallback when no real tokenizer is available (~chars/4 like LangChain)."""

    def __init__(self, ratio: float = 4.0) -> None:
        self._ratio = ratio

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / self._ratio)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return text[: max(1, int(max_tokens * self._ratio))]

    def truncate_tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return text[-max(1, int(max_tokens * self._ratio)) :]


class HFTokenizerCounter:
    def __init__(self, tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def from_file(cls, path: str) -> HFTokenizerCounter:
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))

    @classmethod
    def from_pretrained(cls, name: str) -> HFTokenizerCounter:
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_pretrained(name))

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        enc = self._tok.encode(text)
        if len(enc.ids) <= max_tokens:
            return text
        end_char = enc.offsets[max_tokens - 1][1]
        return text[:end_char]

    def truncate_tail(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        enc = self._tok.encode(text)
        if len(enc.ids) <= max_tokens:
            return text
        start_char = enc.offsets[-max_tokens][0]
        return text[start_char:]


class TiktokenCounter:
    def __init__(self, model: str) -> None:
        import tiktoken

        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        ids = self._enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return self._enc.decode(ids[:max_tokens])

    def truncate_tail(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        ids = self._enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return self._enc.decode(ids[-max_tokens:])


def get_token_counter(cfg: Config) -> TokenCounter:
    src = (getattr(cfg, "tokenizer_source", None) or "auto").lower()
    if src == "char-ratio":
        return CharRatioCounter()

    if src == "auto":
        if cfg.embed_provider == "local":
            tok_path = cfg.local_model_dir / cfg.local_model / "tokenizer.json"
            if tok_path.exists():
                try:
                    return HFTokenizerCounter.from_file(str(tok_path))
                except Exception as exc:
                    logger.debug("HF tokenizer load failed: %s", exc)
        if cfg.embed_provider == "openai":
            try:
                return TiktokenCounter(cfg.openai_model)
            except Exception as exc:
                logger.debug("tiktoken load failed: %s", exc)
        return CharRatioCounter()

    if src == "tiktoken":
        return TiktokenCounter(cfg.openai_model)

    raise ValueError(f"unknown tokenizer_source: {src}")
