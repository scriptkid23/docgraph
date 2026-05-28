import { useCallback, useRef, useState } from "react";
import { postFileWithProgress } from "../lib/upload";
import type { UploadProgressState } from "../types";
import { ProgressBar } from "./ProgressBar";

const idleUpload: UploadProgressState = {
  visible: false,
  percent: 0,
  label: "",
  detail: "",
  failed: false,
};

type UploadVariant = "documents" | "code";

const UPLOAD_VARIANTS: Record<
  UploadVariant,
  {
    title: string;
    intro: string;
    dropLabel: string;
    accept?: string;
    multiple: boolean;
  }
> = {
  documents: {
    title: "Upload",
    intro:
      "Drop PDFs and documents to index for semantic search in Cursor via MCP.",
    dropLabel: "Drop files or click to browse",
    multiple: true,
  },
  code: {
    title: "Code",
    intro:
      "Upload a Repomix codebase dump (.txt or .xml). Each file in the dump is chunked with code-aware boundaries and tagged with its source path.",
    dropLabel: "Drop Repomix dump or click to browse",
    accept: ".txt,.xml,text/plain,application/xml,text/xml",
    multiple: false,
  },
};

interface UploadSectionProps {
  onUploaded: () => void;
  embedded?: boolean;
  variant?: UploadVariant;
}

export function UploadSection({
  onUploaded,
  embedded = false,
  variant = "documents",
}: UploadSectionProps) {
  const config = UPLOAD_VARIANTS[variant];
  const inputRef = useRef<HTMLInputElement>(null);
  const [folder, setFolder] = useState("");
  const [tags, setTags] = useState("");
  const [hover, setHover] = useState(false);
  const [busy, setBusy] = useState(false);
  const [upload, setUpload] = useState<UploadProgressState>(idleUpload);

  const uploadFiles = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (!list.length || busy) return;

      setBusy(true);
      setUpload({
        visible: true,
        percent: 0,
        label: "Preparing",
        detail: `${list.length} file(s) queued`,
        failed: false,
      });

      try {
        for (let i = 0; i < list.length; i++) {
          await postFileWithProgress(
            list[i],
            folder,
            tags,
            i,
            list.length,
            setUpload,
          );
        }
        setUpload({
          visible: true,
          percent: 100,
          label: "Complete",
          detail: "Indexing runs in background — see Status column",
          failed: false,
        });
        onUploaded();
        window.setTimeout(() => setUpload(idleUpload), 1000);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setUpload((u) => ({
          ...u,
          visible: true,
          label: "Failed",
          detail: message,
          failed: true,
        }));
        window.setTimeout(() => setUpload(idleUpload), 4000);
      } finally {
        setBusy(false);
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [busy, folder, tags, onUploaded],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setHover(false);
    if (e.dataTransfer.files.length) void uploadFiles(e.dataTransfer.files);
  };

  const content = (
    <>
      {!embedded && (
        <>
          <p className="label-mono" id="upload-heading">
            01 — Ingest
          </p>
          <h2 className="section-title">{config.title}</h2>
        </>
      )}
      {embedded && (
        <h2 className="section-title ingest-panel-title">{config.title}</h2>
      )}
      <p className="section-intro">{config.intro}</p>

      <div
        className={`dropzone${hover ? " dropzone--hover" : ""}${busy ? " dropzone--disabled" : ""}`}
        onClick={() => !busy && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setHover(true);
        }}
        onDragLeave={() => setHover(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload files"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <span className="dropzone-mark" aria-hidden="true">
          ↑
        </span>
        <span className="dropzone-label">{config.dropLabel}</span>
      </div>

      {upload.visible && (
        <div
          className={`upload-progress${upload.failed ? " upload-progress--failed" : ""}`}
          aria-live="polite"
          role="status"
        >
          <div className="upload-progress-row">
            <span>{upload.label}</span>
            <span className="upload-percent">{Math.round(upload.percent)}%</span>
          </div>
          <ProgressBar percent={upload.percent} inverted={upload.failed} />
          <p className="upload-progress-detail">{upload.detail}</p>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        multiple={config.multiple}
        accept={config.accept}
        hidden
        onChange={(e) => e.target.files && void uploadFiles(e.target.files)}
      />

      <div className="form-row">
        <div className="field">
          <label htmlFor="folder-input">Folder</label>
          <input
            id="folder-input"
            type="text"
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
            placeholder="e.g. DocGraph"
            disabled={busy}
          />
        </div>
        <div className="field">
          <label htmlFor="tags-input">Tags</label>
          <input
            id="tags-input"
            type="text"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="comma-separated"
            disabled={busy}
          />
        </div>
      </div>
    </>
  );

  if (embedded) {
    return content;
  }

  return (
    <section className="section" aria-labelledby="upload-heading">
      {content}
    </section>
  );
}
