export type DocStatus = "processing" | "ready" | "error";

export interface Document {
  id: string;
  filename: string;
  folder: string;
  tags: string[];
  status: DocStatus;
  chunk_count: number;
  progress_pct: number;
  progress_phase: string;
  error_message: string | null;
}

export interface HealthInfo {
  status: string;
  ollama: { ok: boolean; error: string };
  embed_provider: string;
  mcp_sse_url: string;
}

export interface UploadProgressState {
  visible: boolean;
  percent: number;
  label: string;
  detail: string;
  failed: boolean;
}
