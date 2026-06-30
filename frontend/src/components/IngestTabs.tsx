import { useState } from "react";
import { FolderWatchSection } from "./FolderWatchSection";
import { LinkImportSection } from "./LinkImportSection";
import { RepoImportSection } from "./RepoImportSection";
import { UploadSection } from "./UploadSection";

type IngestTab = "upload" | "web" | "code" | "watch" | "repo";

const TABS: { id: IngestTab; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "web", label: "Web pages" },
  { id: "code", label: "Code" },
  { id: "watch", label: "Folder watch" },
  { id: "repo", label: "Repositories" },
];

interface IngestTabsProps {
  onChanged: () => void;
  onReposChanged?: () => void;
}

export function IngestTabs({ onChanged, onReposChanged }: IngestTabsProps) {
  const [tab, setTab] = useState<IngestTab>("upload");

  return (
    <section className="section ingest-section" aria-labelledby="ingest-heading">
      <p className="label-mono" id="ingest-heading">
        01 — Ingest
      </p>

      <div className="ingest-tabs" role="tablist" aria-label="Ingest source">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`ingest-tab-${id}`}
            aria-selected={tab === id}
            aria-controls={`ingest-panel-${id}`}
            className={`ingest-tab${tab === id ? " ingest-tab--active" : ""}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      <div
        role="tabpanel"
        id={`ingest-panel-${tab}`}
        aria-labelledby={`ingest-tab-${tab}`}
        className="ingest-panel"
      >
        {tab === "upload" && (
          <UploadSection embedded onUploaded={onChanged} variant="documents" />
        )}
        {tab === "web" && (
          <LinkImportSection embedded onImported={onChanged} />
        )}
        {tab === "code" && (
          <UploadSection embedded onUploaded={onChanged} variant="code" />
        )}
        {tab === "watch" && <FolderWatchSection onChanged={onChanged} />}
        {tab === "repo" && (
          <RepoImportSection
            embedded
            onImported={onReposChanged || onChanged}
          />
        )}
      </div>
    </section>
  );
}
