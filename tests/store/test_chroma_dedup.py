from docgraph.config import Config
from docgraph.store.chroma import ChromaStore


def test_upsert_skips_duplicate_chunks_within_batch(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    chunks = [
        {
            "id": "d_0",
            "embedding": [0.1] * 768,
            "text": "Welcome to our site",
            "metadata": {
                "doc_id": "d",
                "filename": "a.md",
                "folder": "",
                "tags": "[]",
                "chunk_index": 0,
            },
        },
        {
            "id": "d_1",
            "embedding": [0.1] * 768,
            "text": "Welcome to our site",
            "metadata": {
                "doc_id": "d",
                "filename": "a.md",
                "folder": "",
                "tags": "[]",
                "chunk_index": 1,
            },
        },
        {
            "id": "d_2",
            "embedding": [0.2] * 768,
            "text": "unique content here",
            "metadata": {
                "doc_id": "d",
                "filename": "a.md",
                "folder": "",
                "tags": "[]",
                "chunk_index": 2,
            },
        },
    ]
    store.upsert_chunks(chunks)
    stored = store.get_by_doc_id("d")
    texts = [c["text"] for c in stored]
    assert texts.count("Welcome to our site") == 1
    assert "unique content here" in texts
