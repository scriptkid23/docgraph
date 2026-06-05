from __future__ import annotations

import pytest

from docgraph.mcp.search import FusedHit, _rrf_fuse


def _vec(chunk_id, score=0.5, text="t"):
    return {
        "id": chunk_id,
        "text": text,
        "doc_id": chunk_id.rsplit("_", 1)[0],
        "filename": "f.md",
        "folder": "",
        "tags": [],
        "chunk_index": 0,
        "score": score,
        "source_page": None,
    }


def _sparse(chunk_id, bm25=2.0):
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.rsplit("_", 1)[0],
        "folder": "",
        "tags": [],
        "chunk_index": 0,
        "bm25_score": bm25,
    }


def test_rrf_single_branch_only_vector():
    vec = [_vec("a_0", 0.9), _vec("b_0", 0.8)]
    sparse = []
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    assert out[0].chunk_id == "a_0"
    assert out[0].rrf_score == pytest.approx(1 / 61)
    assert out[1].chunk_id == "b_0"
    assert out[1].rrf_score == pytest.approx(1 / 62)
    assert out[0].vector_score == 0.9
    assert out[0].bm25_score is None


def test_rrf_single_branch_only_sparse():
    vec = []
    sparse = [_sparse("a_0", 5.0), _sparse("b_0", 4.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    assert out[0].chunk_id == "a_0"
    assert out[0].rrf_score == pytest.approx(1 / 61)
    assert out[0].vector_score is None
    assert out[0].bm25_score == 5.0


def test_rrf_consensus_double_counts():
    vec = [_vec("a_0", 0.9), _vec("b_0", 0.8)]
    sparse = [_sparse("a_0", 5.0), _sparse("c_0", 3.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    # 'a_0' is rank 0 in both — gets both reciprocals
    assert out[0].chunk_id == "a_0"
    assert out[0].rrf_score == pytest.approx(2 / 61)
    # Both 'b_0' (vec rank 1) and 'c_0' (sparse rank 1) get 1/62
    by_id = {h.chunk_id: h for h in out}
    assert by_id["b_0"].rrf_score == pytest.approx(1 / 62)
    assert by_id["c_0"].rrf_score == pytest.approx(1 / 62)


def test_rrf_score_propagation_preserves_branch_scores():
    vec = [_vec("a_0", 0.9, text="vector text")]
    sparse = [_sparse("a_0", 5.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    assert out[0].vector_score == 0.9
    assert out[0].bm25_score == 5.0
    assert out[0].text == "vector text"


def test_rrf_sorted_desc_by_score():
    vec = [_vec(f"c_{i}", 0.9 - i * 0.1) for i in range(5)]
    out = _rrf_fuse(vec, [], k_rrf=60)
    scores = [h.rrf_score for h in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_both_returns_empty():
    assert _rrf_fuse([], [], k_rrf=60) == []


def test_rrf_uses_provided_k_rrf():
    # Guards against hardcoded 60: with k=10, rank 0 should yield 1/11.
    vec = [_vec("a_0", 0.9)]
    out = _rrf_fuse(vec, [], k_rrf=10)
    assert out[0].rrf_score == pytest.approx(1 / 11)


def test_rrf_duplicate_within_branch_accumulates():
    # Same chunk_id at rank 0 and rank 3 in vector branch should sum reciprocals.
    vec = [_vec("a_0", 0.9), _vec("b_0", 0.8), _vec("c_0", 0.7),
           _vec("a_0", 0.6)]
    out = _rrf_fuse(vec, [], k_rrf=60)
    by_id = {h.chunk_id: h for h in out}
    # 1/(60+0+1) + 1/(60+3+1) = 1/61 + 1/64
    assert by_id["a_0"].rrf_score == pytest.approx(1 / 61 + 1 / 64)
    # First-seen wins for metadata: vector_score should be 0.9 (rank 0), not 0.6
    assert by_id["a_0"].vector_score == 0.9


def test_rrf_stable_tiebreak_by_chunk_id():
    # When two hits tie on rrf_score, chunk_id ascending decides order.
    vec = [_vec("z_0", 0.5)]
    sparse = [_sparse("a_0", 2.0)]
    out = _rrf_fuse(vec, sparse, k_rrf=60)
    # Both at rank 0 → both have rrf_score = 1/61. Tiebreak by chunk_id asc.
    assert [h.chunk_id for h in out] == ["a_0", "z_0"]
