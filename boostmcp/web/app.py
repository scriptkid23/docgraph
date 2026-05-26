from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles

from boostmcp.config import Config
from boostmcp.models import DocumentRecord, DocumentStatus
from boostmcp.web.deps import AppState

STATIC_DIR = Path(__file__).parent / "static"


def _doc_to_json(doc: DocumentRecord) -> dict:
    return {
        "id": doc.id,
        "filename": doc.filename,
        "folder": doc.folder,
        "tags": doc.tags,
        "status": doc.status.value,
        "chunk_count": doc.chunk_count,
        "error_message": doc.error_message,
    }


async def _run_index(state: AppState, doc_id: str, original_path: Path) -> None:
    try:
        await state.indexer().index_document(doc_id, original_path)
    except Exception:
        pass


def create_app(cfg: Config) -> FastAPI:
    state = AppState.create(cfg)
    app = FastAPI(title="BoostMCP", version="2.0.0")
    app.state.boostmcp = state

    @app.get("/api/health")
    async def health(request: Request):
        st: AppState = request.app.state.boostmcp
        ollama_ok = True
        ollama_error = ""
        try:
            await st.embedder.health_check()
        except Exception as exc:
            ollama_ok = False
            ollama_error = str(exc)
        return {
            "status": "ok",
            "ollama": {"ok": ollama_ok, "error": ollama_error},
            "embed_provider": st.cfg.embed_provider,
        }

    @app.get("/api/documents")
    async def list_documents(request: Request, folder: str | None = None):
        st: AppState = request.app.state.boostmcp
        docs = st.sqlite.list_documents(folder=folder)
        return [_doc_to_json(d) for d in docs]

    @app.post("/api/documents", status_code=202)
    async def upload_document(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        folder: str = Form(""),
        tags: str = Form(""),
    ):
        st: AppState = request.app.state.boostmcp
        content = await file.read()
        max_bytes = st.cfg.max_file_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        if not file.filename:
            raise HTTPException(status_code=415, detail="filename required")

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        orig_path = st.files.save_original(doc_id, file.filename, content)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        doc = DocumentRecord(
            id=doc_id,
            filename=file.filename,
            folder=folder,
            tags=tag_list,
            original_path=str(orig_path),
        )
        st.sqlite.insert_document(doc)
        background_tasks.add_task(_run_index, st, doc_id, orig_path)
        return {"doc_id": doc_id, "status": "processing"}

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(request: Request, doc_id: str):
        st: AppState = request.app.state.boostmcp
        doc = st.sqlite.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="not found")
        st.chroma.delete_by_doc_id(doc_id)
        st.files.delete_doc_files(doc_id)
        st.sqlite.delete_document(doc_id)
        return {"deleted": doc_id}

    @app.patch("/api/documents/{doc_id}")
    async def update_document(
        request: Request,
        doc_id: str,
        folder: str = Form(""),
        tags: str = Form(""),
    ):
        st: AppState = request.app.state.boostmcp
        doc = st.sqlite.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="not found")
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        st.sqlite.update_tags_folder(doc_id, tag_list, folder)
        return {"doc_id": doc_id, "folder": folder, "tags": tag_list}

    @app.post("/api/documents/{doc_id}/reindex", status_code=202)
    async def reindex_document(
        request: Request,
        doc_id: str,
        background_tasks: BackgroundTasks,
    ):
        st: AppState = request.app.state.boostmcp
        doc = st.sqlite.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="not found")
        background_tasks.add_task(st.indexer().reindex_document, doc_id)
        return {"doc_id": doc_id, "status": "processing"}

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
