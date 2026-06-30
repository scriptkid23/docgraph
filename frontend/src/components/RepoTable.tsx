import { useState } from "react";
import { deleteRepo, reindexRepo } from "../api";
import type { Repo } from "../types";
import { ProgressBar } from "./ProgressBar";
import { Button } from "./ui/Button";

interface RepoTableProps {
  repos: Repo[];
  loading: boolean;
  onChanged: () => void;
}

export function RepoTable({ repos, loading, onChanged }: RepoTableProps) {
  const [actingId, setActingId] = useState<string | null>(null);

  const handleReindex = async (id: string) => {
    setActingId(id);
    try {
      await reindexRepo(id);
      onChanged();
    } finally {
      setActingId(null);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!window.confirm(`Delete repo "${name}" and its indexed docs?`)) return;
    setActingId(id);
    try {
      await deleteRepo(id);
      onChanged();
    } finally {
      setActingId(null);
    }
  };

  return (
    <section className="section" aria-labelledby="repo-table-heading">
      <div className="section-head">
        <div>
          <p className="label-mono" id="repo-table-heading">
            04 — Repositories
          </p>
          <h2 className="section-title">Imported repos</h2>
        </div>
        {loading && <span className="muted">Refreshing</span>}
      </div>

      {repos.length === 0 ? (
        <p className="empty-state">
          No repositories imported yet. Paste a GitHub URL or a local path above.
        </p>
      ) : (
        <div className="table-panel">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Repo</th>
                  <th scope="col">Source</th>
                  <th scope="col">Status</th>
                  <th scope="col">Progress</th>
                  <th scope="col">Docs</th>
                  <th scope="col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {repos.map((r) => (
                  <tr key={r.id}>
                    <td className="filename-cell">
                      <div>{r.name}</div>
                      <div className="muted small">{r.id}</div>
                    </td>
                    <td className="filename-cell">
                      <span title={r.source_url || r.local_path}>
                        {r.source_url || r.local_path}
                      </span>
                    </td>
                    <td>
                      <span className={`status-${r.status}`}>{r.status}</span>
                      {r.error_message && (
                        <div className="muted small" title={r.error_message}>
                          {r.error_message.slice(0, 60)}
                        </div>
                      )}
                    </td>
                    <td>
                      {r.status === "processing" ? (
                        <>
                          <ProgressBar percent={r.progress_pct} />
                          <div className="muted small">{r.progress_phase}</div>
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="num-cell">{r.doc_count}</td>
                    <td>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={
                          actingId === r.id || r.status === "processing"
                        }
                        onClick={() => void handleReindex(r.id)}
                      >
                        Re-index
                      </Button>{" "}
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={actingId === r.id}
                        onClick={() => void handleDelete(r.id, r.name)}
                      >
                        Delete
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
