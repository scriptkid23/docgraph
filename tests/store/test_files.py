from pathlib import Path

from boostmcp.config import Config
from boostmcp.store.files import FileStore, sanitize_filename


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("my doc (1).pdf") == "my_doc__1_.pdf"


def test_save_original_and_markdown(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = FileStore(cfg)
    orig = store.save_original("doc_1", "report.pdf", b"PDF bytes")
    md = store.save_markdown("doc_1", "# Title\n\nBody")
    assert orig.exists()
    assert md.exists()
    assert store.read_markdown("doc_1") == "# Title\n\nBody"
    store.delete_doc_files("doc_1")
    assert not orig.exists()
    assert not md.exists()
