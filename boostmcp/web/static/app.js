const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const tbody = document.querySelector("#doc-table tbody");

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

async function uploadFiles(files) {
  const folder = document.getElementById("folder").value;
  const tags = document.getElementById("tags").value;
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("folder", folder);
    fd.append("tags", tags);
    await fetch("/api/documents", { method: "POST", body: fd });
  }
  await loadDocs();
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("hover");
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
