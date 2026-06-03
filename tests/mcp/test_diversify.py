import math

from docgraph.mcp.diversify import mmr_select


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def test_mmr_picks_diverse_results():
    q = _norm([1.0, 0.0])
    cands = [
        {"id": "a", "embedding": _norm([0.95, 0.05]), "score": 0.95},
        {"id": "b", "embedding": _norm([0.94, 0.04]), "score": 0.94},
        {"id": "c", "embedding": _norm([0.93, 0.03]), "score": 0.93},
        {"id": "d", "embedding": _norm([0.5, 0.5]), "score": 0.50},
    ]
    chosen = mmr_select(q, cands, k=2, lambda_=0.3)
    ids = [c["id"] for c in chosen]
    assert ids[0] == "a"
    assert ids[1] == "d"
