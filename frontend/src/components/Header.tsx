import type { HealthInfo } from "../types";
import { Button } from "./ui/Button";

interface HeaderProps {
  health: HealthInfo | null;
  healthError: string | null;
  documentCount: number;
  processingCount: number;
}

export function Header({
  health,
  healthError,
  documentCount,
  processingCount,
}: HeaderProps) {
  let statusText = "Checking Ollama connection…";
  let stripClass = "health-strip health-strip--pending";

  if (healthError) {
    statusText = healthError;
    stripClass = "health-strip health-strip--bad";
  } else if (health) {
    if (health.ollama.ok) {
      statusText = `Ollama operational · ${health.embed_provider}`;
      stripClass = "health-strip health-strip--ok";
    } else {
      statusText = health.ollama.error || "Ollama unavailable";
      stripClass = "health-strip health-strip--bad";
    }
  }

  return (
    <header className="app-header">
      <div>
        <p className="hero-eyebrow">Local document RAG</p>
        <h1 className="hero-title">Documents</h1>
        <div className="hero-rule-wrap" aria-hidden="true">
          <div className="hero-rule" />
        </div>
        <p className={stripClass}>{statusText}</p>
      </div>
      <div className="header-actions">
        <span className="doc-count-pill">
          {documentCount} indexed
          {processingCount > 0 ? ` · ${processingCount} processing` : ""}
        </span>
        {health?.mcp_sse_url && (
          <Button asLink href={health.mcp_sse_url} variant="outline" size="sm">
            MCP SSE →
          </Button>
        )}
      </div>
    </header>
  );
}
