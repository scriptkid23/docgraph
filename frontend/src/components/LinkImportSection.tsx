import { useCallback, useState } from "react";
import { importUrls } from "../api";
import type { UploadProgressState } from "../types";
import { ProgressBar } from "./ProgressBar";
import { Button } from "./ui/Button";

const idleImport: UploadProgressState = {
  visible: false,
  percent: 0,
  label: "",
  detail: "",
  failed: false,
};

interface LinkImportSectionProps {
  onImported: () => void;
  embedded?: boolean;
}

export function LinkImportSection({
  onImported,
  embedded = false,
}: LinkImportSectionProps) {
  const [urls, setUrls] = useState("");
  const [folder, setFolder] = useState("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<UploadProgressState>(idleImport);

  const handleImport = useCallback(async () => {
    if (busy || !urls.trim()) return;

    setBusy(true);
    setProgress({
      visible: true,
      percent: 10,
      label: "Queuing URLs",
      detail: "Submitting import request…",
      failed: false,
    });

    try {
      const result = await importUrls(urls, folder, tags);
      setProgress({
        visible: true,
        percent: 100,
        label: "Queued",
        detail: `${result.queued} URL(s) — crawling runs in background`,
        failed: false,
      });
      setUrls("");
      onImported();
      window.setTimeout(() => setProgress(idleImport), 1500);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Import failed";
      setProgress({
        visible: true,
        percent: 0,
        label: "Failed",
        detail: message,
        failed: true,
      });
      window.setTimeout(() => setProgress(idleImport), 4000);
    } finally {
      setBusy(false);
    }
  }, [busy, urls, folder, tags, onImported]);

  const content = (
    <>
      {!embedded && (
        <>
          <p className="label-mono" id="links-heading">
            02 — Import links
          </p>
          <h2 className="section-title">Web pages</h2>
        </>
      )}
      {embedded && (
        <h2 className="section-title ingest-panel-title">Web pages</h2>
      )}
      <p className="section-intro">
        Paste one URL per line. Pages are fetched with crawl4ai, converted to
        markdown, then chunked and embedded like uploaded files.
      </p>

      <div className="field">
        <label htmlFor="urls-input">URLs</label>
        <textarea
          id="urls-input"
          className="url-textarea"
          value={urls}
          onChange={(e) => setUrls(e.target.value)}
          placeholder={"https://example.com/docs\nhttps://example.com/guide"}
          rows={6}
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
          <label htmlFor="link-folder-input">Folder</label>
          <input
            id="link-folder-input"
            type="text"
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            placeholder="e.g. web-docs"
            disabled={busy}
          />
        </div>
        <div className="field">
          <label htmlFor="link-tags-input">Tags</label>
          <input
            id="link-tags-input"
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
          disabled={busy || !urls.trim()}
          onClick={() => void handleImport()}
        >
          Import URLs
        </Button>
      </div>
    </>
  );

  if (embedded) {
    return content;
  }

  return (
    <section className="section" aria-labelledby="links-heading">
      {content}
    </section>
  );
}
