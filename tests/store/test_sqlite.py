from boostmcp.config import Config
from boostmcp.models import DocumentRecord, DocumentStatus
from boostmcp.store.sqlite import SQLiteStore


def test_insert_and_get_document(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    doc = DocumentRecord(
        id="doc_1",
        filename="a.md",
        folder="proj",
        tags=["tag1"],
        original_path="/tmp/a.md",
    )
    db.insert_document(doc)
    got = db.get_document("doc_1")
    assert got is not None
    assert got.filename == "a.md"
    assert got.folder == "proj"
    assert got.tags == ["tag1"]


def test_list_documents_filter_by_folder(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    db.insert_document(DocumentRecord(id="d1", filename="a.md", folder="A"))
    db.insert_document(DocumentRecord(id="d2", filename="b.md", folder="B"))
    rows = db.list_documents(folder="A")
    assert len(rows) == 1
    assert rows[0].id == "d1"


def test_update_status(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    db.insert_document(DocumentRecord(id="d1", filename="a.md"))
    db.update_status("d1", DocumentStatus.READY, chunk_count=5)
    got = db.get_document("d1")
    assert got.status == DocumentStatus.READY
    assert got.chunk_count == 5
