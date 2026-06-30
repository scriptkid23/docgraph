from docgraph.config import Config
from docgraph.models import DocumentRecord, DocumentStatus, RepoRecord
from docgraph.store.sqlite import SQLiteStore


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


def test_update_progress(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    db.insert_document(DocumentRecord(id="d1", filename="a.md"))
    db.update_progress("d1", 72, "Embedding 18/25 chunks (72%)")
    got = db.get_document("d1")
    assert got.progress_pct == 72
    assert "Embedding" in got.progress_phase


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


def test_repos_lifecycle(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    r = RepoRecord(
        id="repo_a", name="go-ethereum",
        source_url="https://github.com/ethereum/go-ethereum",
        local_path=str(tmp_data_dir / "repos" / "ethereum_go-ethereum"),
        folder="chains", tags=["evm", "core"],
    )
    db.insert_repo(r)
    got = db.get_repo("repo_a")
    assert got is not None
    assert got.name == "go-ethereum"
    assert got.tags == ["evm", "core"]
    assert got.cancel_requested is False
    assert db.get_repo_by_name("go-ethereum").id == "repo_a"
    assert db.get_repo_by_source(
        "https://github.com/ethereum/go-ethereum"
    ).id == "repo_a"
    db.update_repo_progress("repo_a", 40, "Building code index")
    assert db.get_repo("repo_a").progress_phase == "Building code index"
    db.update_repo_status("repo_a", DocumentStatus.READY, doc_count=12)
    g = db.get_repo("repo_a")
    assert g.status == DocumentStatus.READY
    assert g.doc_count == 12
    assert g.progress_pct == 100
    db.update_repo_cancel("repo_a", True)
    assert db.get_repo("repo_a").cancel_requested is True
    db.delete_repo("repo_a")
    assert db.get_repo("repo_a") is None


def test_documents_have_repo_id(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    db = SQLiteStore(cfg)
    db.init_schema()
    doc = DocumentRecord(id="doc_1", filename="README.md", repo_id="repo_a")
    db.insert_document(doc)
    got = db.get_document("doc_1")
    assert got.repo_id == "repo_a"
    by_repo = db.list_documents_by_repo("repo_a")
    assert [d.id for d in by_repo] == ["doc_1"]
