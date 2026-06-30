from docgraph.models import ChunkRecord, DocumentRecord, DocumentStatus, RepoRecord, SearchResult


def test_document_record_defaults():
    doc = DocumentRecord(
        id="doc_abc",
        filename="spec.pdf",
        folder="DocGraph",
        tags=["design"],
    )
    assert doc.status == DocumentStatus.PROCESSING
    assert doc.chunk_count == 0
    assert doc.error_message is None


def test_search_result_fields():
    r = SearchResult(
        text="hello",
        doc_id="doc_abc",
        filename="spec.pdf",
        folder="DocGraph",
        tags=["design"],
        chunk_index=2,
        score=0.87,
    )
    assert r.source_page is None


def test_search_result_file_path_defaults_none():
    r = SearchResult(
        text="x", doc_id="d", filename="f", folder="",
        tags=[], chunk_index=0, score=1.0,
    )
    assert r.file_path is None


def test_chunk_record_accepts_file_path():
    c = ChunkRecord(
        id="d_0", doc_id="d", text="code", chunk_index=0,
        filename="dump.txt", folder="", tags=[], file_path="src/a.py",
    )
    assert c.file_path == "src/a.py"


def test_repo_record_defaults():
    r = RepoRecord(id="repo_x", name="go-ethereum", local_path="/tmp/x")
    assert r.source_url == ""
    assert r.status == DocumentStatus.PROCESSING
    assert r.progress_pct == 0
    assert r.progress_phase == ""
    assert r.error_message is None
    assert r.folder == ""
    assert r.tags == []
    assert r.doc_count == 0
    assert r.cancel_requested is False


def test_document_record_has_repo_id_default():
    d = DocumentRecord(id="doc_1", filename="a.md")
    assert d.repo_id == ""
