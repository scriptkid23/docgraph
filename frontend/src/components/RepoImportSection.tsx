import { useCallback, useState } from "react";
import { importRepo } from "../api";
import type { UploadProgressState } from "../types";
import { ProgressBar } from "./ProgressBar";
import { Button } from "./ui/Button";

const idle: UploadProgressState = {
  visible: false,
  percent: 0,
  label: "",
  detail: "",
  failed: false,
};

interface RepoImportSectionProps {
  onImported: () => void;
  embedded?: boolean;
}

export function RepoImportSection({
  onImported,
  embedded = false,
}: RepoImportSectionProps) {
  const [source, setSource] = useState("");
  const [folder, setFolder] = useState("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<UploadProgressState>(idle);

  const handleImport = useCallback(async () => {
    if (busy || !source.trim()) return;
    setBusy(true);
    setProgress({
      visible: true,
      percent: 10,
      label: "Queuing repo",
      detail: "Submitting import request…",
      failed: false,
    });
    try {
      const result = await importRepo(source.trim(), folder.trim(), tags.trim());
      setProgress({
        visible: true,
        percent: 100,
        label: "Queued",
        detail: `repo_id=${result.repo_id} — cloning + codegraph init in background`,
        failed: false,
      });
      setSource("");
      onImported();
      window.setTimeout(() => setProgress(idle), 2500);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Import failed";
      setProgress({
        visible: true,
        percent: 0,
        label: "Failed",
        detail: message,
        failed: true,
      });
      window.setTimeout(() => setProgress(idle), 4000);
    } finally {
      setBusy(false);
    }
  }, [busy, source, folder, tags, onImported]);

  const content = (
    <>
      {!embedded && (
        <>
          <p className="label-mono" id="repos-heading">
            03 — Import repositories
          </p>
          <h2 className="section-title">Repositories</h2>
        </>
      )}
      {embedded && (
        <h2 className="section-title ingest-panel-title">Repositories</h2>
      )}
      <p className="section-intro">
        Paste a GitHub URL or an absolute local path. Code is indexed via
        codegraph; <code>*.md</code> files are vectorized through the existing
        pipeline.
      </p>

      <div className="field">
        <label htmlFor="repo-source-input">Source</label>
        <input
          id="repo-source-input"
          type="text"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="https://github.com/ethereum/go-ethereum"
          disabled={busy}
        />
      </div>

      {progress.visible && (
        <div
          className={`upload-progress${progress.failed ? " upload-progress--failed" : ""}`}
          aria-live="polite"
          role="status"
        >
          <div className="upload-progress-row">
            <span>{progress.label}</span>
            <span className="upload-percent">{Math.round(progress.percent)}%</span>
          </div>
          <ProgressBar percent={progress.percent} inverted={progress.failed} />
          <p className="upload-progress-detail">{progress.detail}</p>
        </div>
      )}

      <div className="form-row">
        <div className="field">
          <label htmlFor="repo-folder-input">Folder</label>
          <input
            id="repo-folder-input"
            type="text"
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            placeholder="e.g. chains"
            disabled={busy}
          />
        </div>
        <div className="field">
          <label htmlFor="repo-tags-input">Tags</label>
          <input
            id="repo-tags-input"
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="comma-separated"
            disabled={busy}
          />
        </div>
      </div>

      <div className="import-actions">
        <Button
          variant="primary"
          disabled={busy || !source.trim()}
          onClick={() => void handleImport()}
        >
          Import repo
        </Button>
      </div>
    </>
  );

  if (embedded) {
    return content;
  }

  return (
    <section className="section" aria-labelledby="repos-heading">
      {content}
    </section>
  );
}
