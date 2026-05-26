const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const tbody = document.querySelector("#doc-table tbody");
const uploadProgress = document.getElementById("upload-progress");
const uploadLabel = document.getElementById("upload-label");
const uploadPercent = document.getElementById("upload-percent");
const uploadBar = document.getElementById("upload-bar");
const uploadDetail = document.getElementById("upload-detail");

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

async function loadDocs() {
  const resp = await fetch("/api/documents");
  const docs = await resp.json();
  tbody.innerHTML = docs.map((d) => `
    <tr>
      <td>${escapeHtml(d.filename)}</td>
      <td>${escapeHtml(d.folder || "-")}</td>
      <td>${escapeHtml((d.tags || []).join(", "))}</td>
      <td class="status-${d.status}">${escapeHtml(d.status)}${d.error_message ? ": " + escapeHtml(d.error_message) : ""}</td>
      <td>${d.chunk_count}</td>
      <td>
        <button onclick="reindex('${d.id}')">Re-index</button>
        <button onclick="removeDoc('${d.id}')">Delete</button>
      </td>
    </tr>
  `).join("");
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
    const basePercent = (fileIndex / totalFiles) * 100;
    const slice = 100 / totalFiles;

    xhr.upload.addEventListener("progress", (e) => {
      let filePct = 50;
      if (e.lengthComputable && e.total > 0) {
        filePct = (e.loaded / e.total) * 100;
      }
      const overall = basePercent + (filePct / 100) * slice;
      const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
      const totalMb = e.lengthComputable ? (e.total / (1024 * 1024)).toFixed(1) : "?";
      setUploadUI(
        true,
        overall,
        `Uploading ${file.name}`,
        `File ${fileIndex + 1} of ${totalFiles} · ${loadedMb} / ${totalMb} MB`,
      );
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
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
      const donePct = ((i + 1) / list.length) * 100;
      setUploadUI(
        true,
        donePct,
        i + 1 < list.length ? "Upload complete, next file…" : "Upload complete",
        `File ${i + 1} of ${list.length} finished`,
      );
    }
    setUploadUI(true, 100, "Done", "Refreshing document list…");
    await loadDocs();
    setTimeout(() => setUploadUI(false, 0, "", ""), 1200);
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

loadHealth();
loadDocs();
setInterval(loadDocs, 5000);
