const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const tbody = document.querySelector("#doc-table tbody");
const uploadProgress = document.getElementById("upload-progress");
const uploadLabel = document.getElementById("upload-label");
const uploadPercent = document.getElementById("upload-percent");
const uploadBar = document.getElementById("upload-bar");
const uploadDetail = document.getElementById("upload-detail");

const UPLOAD_PROGRESS_CAP = 88;
const UPLOAD_SERVER_PHASE = 94;

let uploadInProgress = false;
let lastUploadPercent = 0;

async function loadHealth() {
  const resp = await fetch("/api/health");
  const data = await resp.json();
  const el = document.getElementById("health");
  el.textContent = data.ollama.ok
    ? `Ollama OK (${data.embed_provider})`
    : `Ollama: ${data.ollama.error}`;
}

function formatStatusCell(d) {
  if (d.status === "processing") {
    const pct = Number(d.progress_pct) || 0;
    const phase = d.progress_phase || "Indexing…";
    return `
      <div class="status-processing-wrap">
        <div class="status-processing-title">${escapeHtml(phase)}</div>
        <div class="doc-progress-track" aria-hidden="true">
          <div class="doc-progress-bar" style="width:${pct}%"></div>
        </div>
      </div>`;
  }
  if (d.status === "error" && d.error_message) {
    return `<span class="status-error">${escapeHtml(d.error_message)}</span>`;
  }
  return `<span class="status-${d.status}">${escapeHtml(d.status)}</span>`;
}

async function loadDocs() {
  const resp = await fetch("/api/documents");
  const docs = await resp.json();
  tbody.innerHTML = docs.map((d) => `
    <tr>
      <td>${escapeHtml(d.filename)}</td>
      <td>${escapeHtml(d.folder || "-")}</td>
      <td>${escapeHtml((d.tags || []).join(", "))}</td>
      <td>${formatStatusCell(d)}</td>
      <td>${d.status === "processing" ? "…" : d.chunk_count}</td>
      <td>
        <button onclick="reindex('${d.id}')">Re-index</button>
        <button onclick="removeDoc('${d.id}')">Delete</button>
      </td>
    </tr>
  `).join("");
  return docs;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function setUploadUI(visible, percent, label, detail) {
  uploadProgress.classList.toggle("hidden", !visible);
  const pct = Math.min(100, Math.max(0, Math.round(percent)));
  lastUploadPercent = pct;
  uploadPercent.textContent = `${pct}%`;
  uploadBar.style.width = `${pct}%`;
  uploadLabel.textContent = label;
  uploadDetail.textContent = detail;
}

function hideUploadProgressSoon(delayMs = 900) {
  setTimeout(() => setUploadUI(false, 0, "", ""), delayMs);
}

function setUploadBusy(busy) {
  uploadInProgress = busy;
  dropzone.classList.toggle("disabled", busy);
  fileInput.disabled = busy;
}

function postFileWithProgress(file, folder, tags, fileIndex, totalFiles) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("folder", folder);
    fd.append("tags", tags);

    const xhr = new XMLHttpRequest();
    xhr.timeout = 600000;
    const basePercent = (fileIndex / totalFiles) * 100;
    const slice = 100 / totalFiles;
    let bytesFullySent = false;

    xhr.upload.addEventListener("progress", (e) => {
      const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
      const totalMb = e.lengthComputable ? (e.total / (1024 * 1024)).toFixed(1) : "?";

      if (e.lengthComputable && e.total > 0 && e.loaded >= e.total) {
        bytesFullySent = true;
        const overall = basePercent + (UPLOAD_SERVER_PHASE / 100) * slice;
        setUploadUI(
          true,
          overall,
          `Saving ${file.name}…`,
          `File ${fileIndex + 1} of ${totalFiles} · sent ${totalMb} MB, waiting for server`,
        );
        return;
      }

      let filePct = 50;
      if (e.lengthComputable && e.total > 0) {
        filePct = Math.min(UPLOAD_PROGRESS_CAP, (e.loaded / e.total) * 100);
      }
      const overall = basePercent + (filePct / 100) * slice;
      setUploadUI(
        true,
        overall,
        `Uploading ${file.name}`,
        `File ${fileIndex + 1} of ${totalFiles} · ${loadedMb} / ${totalMb} MB`,
      );
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const overall = basePercent + slice;
        setUploadUI(
          true,
          overall,
          `Saved ${file.name}`,
          bytesFullySent
            ? "Upload accepted — indexing runs in background"
            : `File ${fileIndex + 1} of ${totalFiles} saved`,
        );
        resolve();
        return;
      }
      let msg = `Upload failed (${xhr.status})`;
      try {
        const body = JSON.parse(xhr.responseText);
        if (body.detail) msg = body.detail;
      } catch (_) {
        /* ignore */
      }
      reject(new Error(msg));
    });

    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));
    xhr.addEventListener("timeout", () => reject(new Error("Upload timed out (server too slow)")));

    xhr.open("POST", "/api/documents");
    xhr.send(fd);
  });
}

async function uploadFiles(files) {
  const list = Array.from(files);
  if (!list.length || uploadInProgress) return;

  setUploadBusy(true);
  setUploadUI(true, 0, "Preparing upload…", `${list.length} file(s)`);

  const folder = document.getElementById("folder").value;
  const tags = document.getElementById("tags").value;

  try {
    for (let i = 0; i < list.length; i++) {
      await postFileWithProgress(list[i], folder, tags, i, list.length);
    }
    setUploadUI(
      true,
      100,
      "Upload complete",
      "Indexing in background — status updates every few seconds",
    );
    void loadDocs().catch(() => {});
    hideUploadProgressSoon(1000);
  } catch (err) {
    setUploadUI(true, lastUploadPercent, "Upload failed", err.message);
    uploadBar.style.background = "#dc3545";
    setTimeout(() => {
      uploadBar.style.background = "";
      setUploadUI(false, 0, "", "");
    }, 4000);
  } finally {
    setUploadBusy(false);
    fileInput.value = "";
  }
}

dropzone.addEventListener("click", () => {
  if (!uploadInProgress) fileInput.click();
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (!uploadInProgress) dropzone.classList.add("hover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("hover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("hover");
  uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => uploadFiles(fileInput.files));

async function removeDoc(id) {
  await fetch(`/api/documents/${id}`, { method: "DELETE" });
  await loadDocs();
}

async function reindex(id) {
  await fetch(`/api/documents/${id}/reindex`, { method: "POST" });
  await loadDocs();
}

let docPollTimer = null;

function scheduleDocPoll() {
  if (docPollTimer) clearInterval(docPollTimer);
  const tick = async () => {
    const docs = await loadDocs().catch(() => []);
    const hasProcessing = docs.some((d) => d.status === "processing");
    if (docPollTimer) clearInterval(docPollTimer);
    docPollTimer = setInterval(tick, hasProcessing ? 1500 : 5000);
  };
  tick();
}

loadHealth();
scheduleDocPoll();
