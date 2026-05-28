from docgraph.models import ChunkRecord, DocumentRecord, DocumentStatus, SearchResult


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
