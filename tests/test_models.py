from boostmcp.models import DocumentRecord, DocumentStatus, SearchResult


def test_document_record_defaults():
    doc = DocumentRecord(
        id="doc_abc",
        filename="spec.pdf",
        folder="BoostMCP",
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
        folder="BoostMCP",
        tags=["design"],
        chunk_index=2,
        score=0.87,
    )
    assert r.source_page is None
