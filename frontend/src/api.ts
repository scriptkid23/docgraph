import type { Document, HealthInfo } from "./types";

export async function fetchHealth(): Promise<HealthInfo> {
  const resp = await fetch("/api/health");
  if (!resp.ok) throw new Error("Health check failed");
  return resp.json();
}

export async function fetchDocuments(): Promise<Document[]> {
  const resp = await fetch("/api/documents");
  if (!resp.ok) throw new Error("Failed to load documents");
  return resp.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const resp = await fetch(`/api/documents/${id}`, { method: "DELETE" });
  if (!resp.ok) throw new Error("Delete failed");
}

export async function reindexDocument(id: string): Promise<void> {
  const resp = await fetch(`/api/documents/${id}/reindex`, { method: "POST" });
  if (!resp.ok) throw new Error("Re-index failed");
}

export interface ImportUrlsResult {
  queued: number;
  doc_ids: string[];
}

export async function importUrls(
  urls: string,
  folder: string,
  tags: string,
): Promise<ImportUrlsResult> {
  const resp = await fetch("/api/documents/import-urls", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls, folder, tags }),
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : "Import failed";
    throw new Error(detail);
  }
  return resp.json();
}
