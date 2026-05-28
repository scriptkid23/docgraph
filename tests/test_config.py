import os
from pathlib import Path

import pytest
import yaml

from docgraph.config import Config, load_config, normalize_ollama_url


def test_normalize_ollama_url_localhost():
    assert normalize_ollama_url("http://localhost:11434") == "http://127.0.0.1:11434"


def test_load_config_normalizes_localhost_ollama_url(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(
        yaml.dump({"embedding": {"ollama_url": "http://localhost:11434"}})
    )
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(data_dir))
    cfg = load_config()
    assert cfg.ollama_url == "http://127.0.0.1:11434"


def test_config_defaults(tmp_data_dir):
    cfg = Config(data_dir=tmp_data_dir)
    assert cfg.web_host == "127.0.0.1"
    assert cfg.ollama_url == "http://127.0.0.1:11434"
    assert cfg.web_port == 8080
    assert cfg.embed_provider == "local"
    assert cfg.local_model == "nomic-embed-text"
    assert cfg.chunk_size == 512
    assert cfg.default_top_k == 5


def test_load_config_from_yaml(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg_file = data_dir / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "server": {"web_port": 9090},
        "embedding": {"provider": "openai", "openai_model": "text-embedding-3-small"},
    }))
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(data_dir))
    cfg = load_config()
    assert cfg.web_port == 9090
    assert cfg.embed_provider == "openai"
    assert cfg.openai_model == "text-embedding-3-small"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(yaml.dump({"server": {"web_port": 9090}}))
    monkeypatch.setenv("DOCGRAPH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DOCGRAPH_WEB_PORT", "7777")
    cfg = load_config()
    assert cfg.web_port == 7777
