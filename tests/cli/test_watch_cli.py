from unittest.mock import patch, MagicMock

import pytest

from docgraph.cli_watch import run_watch_command


def _resp(status_code: int, json_body: dict):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    r.raise_for_status = MagicMock()
    return r


def test_enable_command(capsys):
    with patch("docgraph.cli_watch.httpx.Client") as m:
        m.return_value.__enter__.return_value.post.return_value = _resp(
            202, {"enabled": True, "dirs": 0, "reconcile_started": True}
        )
        rc = run_watch_command(["enable"], "http://127.0.0.1:8088")
    assert rc == 0
    out = capsys.readouterr().out
    assert "enabled" in out.lower()


def test_status_command(capsys):
    with patch("docgraph.cli_watch.httpx.Client") as m:
        m.return_value.__enter__.return_value.get.return_value = _resp(200, {
            "enabled": True, "running": True, "dirs_count": 2,
            "queue_depth": 0, "queue_capacity": 500, "workers": 4,
            "last_enabled_at": None, "stats": {
                "events_received": 0, "events_debounced": 0, "events_processed": 0,
                "events_dropped_queue_full": 0, "reconcile_runs": 1, "last_reconcile_at": None,
            },
        })
        rc = run_watch_command(["status"], "http://127.0.0.1:8088")
    assert rc == 0


def test_add_command(capsys, tmp_path):
    with patch("docgraph.cli_watch.httpx.Client") as m:
        m.return_value.__enter__.return_value.post.return_value = _resp(
            201, {"id": "wd_x", "path": str(tmp_path), "scheduled": True}
        )
        rc = run_watch_command(
            ["add", str(tmp_path), "--folder", "notes", "--tags", "a,b"],
            "http://127.0.0.1:8088",
        )
    assert rc == 0


def test_server_unreachable(capsys):
    import httpx
    with patch("docgraph.cli_watch.httpx.Client") as m:
        m.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("refused")
        rc = run_watch_command(["status"], "http://127.0.0.1:8088")
    assert rc == 1
    err = capsys.readouterr().err
    assert "server not running" in err.lower()
