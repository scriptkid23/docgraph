from docgraph.config import Config
from docgraph.store.chroma import ChromaStore


def test_upsert_and_search(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    vec = [0.1] * 768
    store.upsert_chunks([
        {
            "id": "doc_1_0",
            "embedding": vec,
            "text": "Ollama embedding config",
            "metadata": {
                "doc_id": "doc_1",
                "filename": "readme.md",
                "folder": "DocGraph",
                "tags": "design,v2",
                "chunk_index": 0,
            },
        }
    ])
    results = store.search(
        query_embedding=vec,
        top_k=1,
        folder="DocGraph",
    )
    assert len(results) == 1
    assert results[0]["text"] == "Ollama embedding config"
    store.delete_by_doc_id("doc_1")
    assert store.search(query_embedding=vec, top_k=1) == []


def test_search_returns_file_path(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    store.upsert_chunks([{
        "id": "d_0",
        "embedding": [0.1] * 768,
        "text": "def a(): return 1",
        "metadata": {
            "doc_id": "d", "filename": "dump.txt", "folder": "",
            "tags": "[]", "chunk_index": 0, "file_path": "src/a.py",
        },
    }])
    results = store.search(query_embedding=[0.1] * 768, top_k=1)
    assert results[0]["file_path"] == "src/a.py"
