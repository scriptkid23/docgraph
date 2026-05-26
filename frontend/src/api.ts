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
