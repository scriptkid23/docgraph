import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchDocuments, fetchHealth, fetchRepos } from "./api";
import { DocumentTable } from "./components/DocumentTable";
import { Header } from "./components/Header";
import { IngestTabs } from "./components/IngestTabs";
import { RepoTable } from "./components/RepoTable";
import { SectionRule } from "./components/ui/SectionRule";
import type { Document, HealthInfo, Repo } from "./types";

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [repos, setRepos] = useState<Repo[]>([]);
  const [reposLoading, setReposLoading] = useState(false);

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

  const refreshRepos = useCallback(async () => {
    setReposLoading(true);
    try {
      const rs = await fetchRepos();
      setRepos(rs);
      return rs;
    } catch {
      return [];
    } finally {
      setReposLoading(false);
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
    () =>
      documents.filter((d) => d.status === "processing").length +
      repos.filter((r) => r.status === "processing").length,
    [documents, repos],
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
      const [docs, rs] = await Promise.all([refreshDocs(), refreshRepos()]);
      if (cancelled) return;
      const busy =
        docs.some((d) => d.status === "processing") ||
        rs.some((r) => r.status === "processing");
      timer = window.setTimeout(tick, busy ? 1500 : 5000);
    };

    void tick();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [refreshDocs, refreshRepos]);

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
          <IngestTabs
            onChanged={() => void refreshDocs()}
            onReposChanged={() => void refreshRepos()}
          />
          <SectionRule thick />
          <RepoTable
            repos={repos}
            loading={reposLoading}
            onChanged={() => {
              void refreshRepos();
              void refreshDocs();
            }}
          />
          <SectionRule thick />
          <DocumentTable
            documents={documents}
            repos={repos}
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
