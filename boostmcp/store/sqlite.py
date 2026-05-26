from __future__ import annotations

import json
import sqlite3
from typing import Optional

from boostmcp.config import Config
from boostmcp.models import DocumentRecord, DocumentStatus


class SQLiteStore:
    def __init__(self, cfg: Config) -> None:
        self._path = cfg.sqlite_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    folder TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'processing',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    original_path TEXT NOT NULL DEFAULT '',
                    markdown_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)

    def insert_document(self, doc: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, filename, folder, tags, status, chunk_count,
                    error_message, original_path, markdown_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.id, doc.filename, doc.folder, json.dumps(doc.tags),
                    doc.status.value, doc.chunk_count, doc.error_message,
                    doc.original_path, doc.markdown_path,
                ),
            )

    def _row_to_doc(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            folder=row["folder"],
            tags=json.loads(row["tags"]),
            status=DocumentStatus(row["status"]),
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            original_path=row["original_path"],
            markdown_path=row["markdown_path"],
        )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def list_documents(
        self,
        folder: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> list[DocumentRecord]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: list = []
        if folder is not None:
            query += " AND folder = ?"
            params.append(folder)
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        docs = [self._row_to_doc(r) for r in rows]
        if tag:
            docs = [d for d in docs if tag in d.tags]
        return docs

    def update_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None,
        markdown_path: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE documents SET status=?, chunk_count=?,
                   error_message=?, markdown_path=? WHERE id=?""",
                (status.value, chunk_count, error_message, markdown_path, doc_id),
            )

    def update_tags_folder(self, doc_id: str, tags: list[str], folder: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET tags=?, folder=? WHERE id=?",
                (json.dumps(tags), folder, doc_id),
            )

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
