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


def test_count_chunks(tmp_path):
    from docgraph.config import Config
    from docgraph.store.chroma import ChromaStore

    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    store = ChromaStore(cfg)
    assert store.count_chunks() == 0
    store.upsert_chunks([
        {"id": "a_0", "embedding": [0.1] * 768, "text": "x", "metadata": {"doc_id": "a"}},
        {"id": "a_1", "embedding": [0.2] * 768, "text": "y", "metadata": {"doc_id": "a"}},
    ])
    assert store.count_chunks() == 2


def test_update_doc_metadata_changes_folder_and_tags(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    chroma = ChromaStore(cfg)
    chroma.upsert_chunks([{
        "id": "d_0", "embedding": [0.1] * 768, "text": "hello",
        "metadata": {"doc_id": "d", "filename": "f.md", "folder": "old",
                     "tags": '["a"]', "chunk_index": 0},
    }])
    chroma.update_doc_metadata("d", "new", ["b"])
    # Should find in new folder
    hits = chroma.search([0.1] * 768, top_k=10, folder="new")
    assert len(hits) >= 1
    assert hits[0]["folder"] == "new"
    # ChromaStore.search already JSON-decodes tags via _decode_tags
    assert hits[0]["tags"] == ["b"]
    # Old folder should return nothing
    old_hits = chroma.search([0.1] * 768, top_k=10, folder="old")
    assert old_hits == []
