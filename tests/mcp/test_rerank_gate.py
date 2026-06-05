from __future__ import annotations

import pytest

from docgraph.config import Config
from docgraph.mcp.search import FusedHit, _should_rerank


def _hit(rrf=0.03, has_vector=True, has_sparse=True, chunk_id="x"):
    return FusedHit(
        chunk_id=chunk_id,
        text="",
        doc_id="d",
        filename="",
        folder="",
        tags=[],
        chunk_index=0,
        source_page=None,
        vector_score=0.5 if has_vector else None,
        bm25_score=2.0 if has_sparse else None,
        rrf_score=rrf,
    )


@pytest.fixture
def cfg(tmp_path):
    c = Config(data_dir=tmp_path)
    c.rerank_enabled = True
    c.rerank_min_floor = 0.015
    c.rerank_score_gap_ratio = 0.5
    return c


class TestShouldRerank:
    def test_skip_when_rerank_disabled(self, cfg):
        cfg.rerank_enabled = False
        decision, reason = _should_rerank(cfg, [_hit(0.03), _hit(0.02)], k=5)
        assert decision is False
        assert reason == "skip_disabled"

    def test_skip_when_zero_candidates(self, cfg):
        decision, reason = _should_rerank(cfg, [], k=5)
        assert decision is False
        assert reason == "skip_too_few_candidates"

    def test_skip_when_one_candidate(self, cfg):
        decision, reason = _should_rerank(cfg, [_hit(0.03)], k=5)
        assert decision is False
        assert reason == "skip_too_few_candidates"

    def test_skip_when_top1_below_floor(self, cfg):
        cands = [_hit(rrf=0.010, chunk_id="a"), _hit(rrf=0.008, chunk_id="b")]
        decision, reason = _should_rerank(cfg, cands, k=5)
        assert decision is False
        assert reason == "skip_floor"

    def test_force_rerank_when_only_vector_branch(self, cfg):
        cands = [_hit(rrf=0.03, has_sparse=False, chunk_id=f"c{i}") for i in range(3)]
        decision, reason = _should_rerank(cfg, cands, k=5)
        assert decision is True
        assert reason == "force_single_branch"

    def test_force_rerank_when_only_sparse_branch(self, cfg):
        cands = [_hit(rrf=0.03, has_vector=False, chunk_id=f"c{i}") for i in range(3)]
        decision, reason = _should_rerank(cfg, cands, k=5)
        assert decision is True
        assert reason == "force_single_branch"

    def test_skip_when_gap_clear(self, cfg):
        # gap_ratio = (0.030 - 0.005) / 0.030 = 0.833 > 0.5 -> skip
        cands = [
            _hit(rrf=0.030, chunk_id="a"),
            _hit(rrf=0.020, chunk_id="b"),
            _hit(rrf=0.015, chunk_id="c"),
            _hit(rrf=0.010, chunk_id="d"),
            _hit(rrf=0.005, chunk_id="e"),
        ]
        decision, reason = _should_rerank(cfg, cands, k=5)
        assert decision is False
        assert reason == "skip_gap"

    def test_rerank_when_gap_small(self, cfg):
        # gap_ratio = (0.030 - 0.026) / 0.030 = 0.133 < 0.5 -> rerank
        cands = [_hit(rrf=0.030 - i * 0.001, chunk_id=f"c{i}") for i in range(5)]
        decision, reason = _should_rerank(cfg, cands, k=5)
        assert decision is True
        assert reason == "force_ambiguous"

    def test_floor_check_fires_before_zero_division(self, cfg):
        # top1 = 0 -> floor check (0 < 0.015) skips before gap-ratio division.
        cands = [_hit(rrf=0.0, chunk_id="a"), _hit(rrf=0.0, chunk_id="b")]
        decision, reason = _should_rerank(cfg, cands, k=5)
        assert decision is False
        assert reason == "skip_floor"
