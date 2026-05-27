import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchDocuments, fetchHealth } from "./api";
import { DocumentTable } from "./components/DocumentTable";
import { Header } from "./components/Header";
import { UploadSection } from "./components/UploadSection";
import { SectionRule } from "./components/ui/SectionRule";
import type { Document, HealthInfo } from "./types";

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
      return docs;
    } catch {
      return [];
    } finally {
      setDocsLoading(false);
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await fetchHealth());
      setHealthError(null);
    } catch (e) {
      setHealthError(e instanceof Error ? e.message : "Health unavailable");
    }
  }, []);

  const processingCount = useMemo(
    () => documents.filter((d) => d.status === "processing").length,
    [documents],
  );

  useEffect(() => {
    void refreshHealth();
    const id = window.setInterval(() => void refreshHealth(), 30_000);
    return () => window.clearInterval(id);
  }, [refreshHealth]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;

    const tick = async () => {
      if (cancelled) return;
      const docs = await refreshDocs();
      if (cancelled) return;
      const ms = docs.some((d) => d.status === "processing") ? 1500 : 5000;
      timer = window.setTimeout(tick, ms);
    };

    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [refreshDocs]);

  return (
    <div className="page">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <div className="page-texture" aria-hidden="true" />
      <div className="container">
        <Header
          health={health}
          healthError={healthError}
          documentCount={documents.length}
          processingCount={processingCount}
        />
        <SectionRule ultra />
        <main id="main">
          <UploadSection onUploaded={() => void refreshDocs()} />
          <SectionRule thick />
          <DocumentTable
            documents={documents}
            loading={docsLoading}
            onChanged={() => void refreshDocs()}
          />
        </main>
        <SectionRule thick />
        <footer className="label-mono" style={{ paddingTop: "1.5rem" }}>
          DocGraph · Monochrome editorial interface
        </footer>
      </div>
    </div>
  );
}
