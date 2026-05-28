from __future__ import annotations


def uses_e5_prefixes(model: str) -> bool:
    """True for intfloat multilingual-e5 models (query:/passage: prefixes)."""
    normalized = model.lower().replace("_", "").replace("-", "")
    return "multilinguale5" in normalized or normalized.startswith("e5")


def apply_embed_prefix(
    texts: list[str], model: str, *, for_query: bool
) -> list[str]:
    if not uses_e5_prefixes(model):
        return texts
    label = "query" if for_query else "passage"
    prefix = f"{label}: "
    out: list[str] = []
    for text in texts:
        stripped = text.lstrip()
        if stripped.startswith("query:") or stripped.startswith("passage:"):
            out.append(text)
        else:
            out.append(prefix + text)
    return out
