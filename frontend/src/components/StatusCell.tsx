import type { Document } from "../types";
import { ProgressBar } from "./ProgressBar";

interface StatusCellProps {
  doc: Document;
}

export function StatusCell({ doc }: StatusCellProps) {
  if (doc.status === "processing") {
    const pct = doc.progress_pct ?? 0;
    const phase = doc.progress_phase || "Indexing";
    return (
      <div className="status-processing-wrap">
        <div className="status-processing-title">
          {phase} — {pct}%
        </div>
        <ProgressBar percent={pct} />
      </div>
    );
  }

  if (doc.status === "error" && doc.error_message) {
    return (
      <span className="status-tooltip-wrap">
        <span
          className="status-badge status-badge--error"
          tabIndex={0}
          aria-label={`Error: ${doc.error_message}`}
        >
          Error
        </span>
        <span className="status-tooltip" role="tooltip">
          {doc.error_message}
        </span>
      </span>
    );
  }

  if (doc.status === "error") {
    return <span className="status-badge status-badge--error">Error</span>;
  }

  if (doc.status === "ready") {
    return <span className="status-badge status-badge--ready">Ready</span>;
  }

  return <span className="status-badge">{doc.status}</span>;
}
