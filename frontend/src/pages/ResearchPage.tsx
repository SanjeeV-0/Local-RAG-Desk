/**
 * Top-level page and sole owner of cross-tab state: which tab is active,
 * the global Trace Mode toggle (threaded into BOTH the Research tab's
 * /research/ask calls and the Retrieval Explorer's /retrieval/search calls
 * -- one toggle, two consumers, so its displayed state always matches what
 * every request actually sends), the last research response, and
 * `targetDocId` used for cross-tab navigation (e.g. "Inspect Doc" from a
 * retrieval result jumps to the Document Library pre-selected on that doc).
 */
import React, { useState, useEffect } from 'react';
import { Header, WorkspaceTab } from '../components/Header';
import { ResearchInput } from '../components/ResearchInput';
import { AnswerPanel } from '../components/AnswerPanel';
import { SourcesPanel } from '../components/SourcesPanel';
import { TracePanel } from '../components/TracePanel';
import { DownloadButton } from '../components/DownloadButton';
import { LoadingOverlay } from '../components/LoadingOverlay';
import { RetrievalWorkspace } from '../components/RetrievalWorkspace';
import { DocumentWorkspace } from '../components/DocumentWorkspace';
import {
  askResearchQuestion,
  checkBackendHealth,
  checkBackendReadiness,
} from '../api/research';
import {
  ResearchResponse,
  BackendReadinessResponse,
  TraceData,
} from '../types/research';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export const ResearchPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('research');
  const [traceEnabled, setTraceEnabled] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<ResearchResponse | null>(null);
  const [activeTraceData, setActiveTraceData] = useState<TraceData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [readinessInfo, setReadinessInfo] = useState<BackendReadinessResponse | null>(null);
  const [lastQuery, setLastQuery] = useState<string>('');

  // Target document selected for inspection
  const [targetDocId, setTargetDocId] = useState<string | null>(null);

  const checkStatus = async () => {
    const health = await checkBackendHealth();
    setIsBackendHealthy(health !== null && health.status === 'healthy');

    const ready = await checkBackendReadiness();
    setReadinessInfo(ready);
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleResearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setLastQuery(query);

    try {
      const data = await askResearchQuestion(query, traceEnabled);
      setResponse(data);
      setIsBackendHealthy(true);

      if (data.trace) {
        setActiveTraceData(data.trace);
      }
    } catch (err: any) {
      console.error('Research request failed:', err);
      setError(
        err?.message ||
          'An unexpected error occurred while communicating with the research agent.'
      );
      setIsBackendHealthy(false);
      // Clear the previous successful response so it can never be mistaken
      // for the answer/sources belonging to this failed request.
      setResponse(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (lastQuery) {
      handleResearch(lastQuery);
    }
  };

  const handleNavigateToDoc = (docId: string) => {
    setTargetDocId(docId);
    setActiveTab('documents');
  };

  const handleNavigateToRetrieval = (docId: string) => {
    setTargetDocId(docId);
    setActiveTab('retrieval');
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        traceEnabled={traceEnabled}
        onToggleTrace={setTraceEnabled}
        isBackendHealthy={isBackendHealthy}
        readinessInfo={readinessInfo}
        hasTraceData={!!(activeTraceData || response?.trace)}
      />

      <main style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
        {/* ================= TAB 1: RESEARCH & QA ================= */}
        {activeTab === 'research' && (
          <>
            <ResearchInput
              onSearch={handleResearch}
              isLoading={isLoading}
              initialQuery={lastQuery}
            />

            {error && (
              <div className="error-banner">
                <AlertTriangle
                  size={24}
                  style={{ flexShrink: 0, marginTop: 2, color: '#f43f5e' }}
                />
                <div style={{ flex: 1 }}>
                  <h4 className="error-title">Research Execution Error</h4>
                  <p className="error-desc">{error}</p>
                </div>
                {lastQuery && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={handleRetry}
                    style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}
                  >
                    <RefreshCw size={14} /> Retry
                  </button>
                )}
              </div>
            )}

            {isLoading ? (
              <LoadingOverlay traceMode={traceEnabled} />
            ) : (
              <>
                <div className="workspace-grid">
                  <AnswerPanel
                    answer={response?.answer ?? null}
                    model={response?.model ?? null}
                    isLoading={false}
                  />
                  <SourcesPanel
                    sources={response?.sources ?? []}
                    isLoading={false}
                    finalResults={response?.trace?.final_results}
                    onInspectDocument={handleNavigateToDoc}
                  />
                </div>

                {/* Render Download JSON button when a response is present */}
                {response && <DownloadButton response={response} />}

                {/* Render Trace Panel inline when trace is present and enabled */}
                {traceEnabled && response?.trace && (
                  <TracePanel trace={response.trace} />
                )}
              </>
            )}
          </>
        )}

        {/* ================= TAB 2: VECTOR RETRIEVAL EXPLORER ================= */}
        {activeTab === 'retrieval' && (
          <RetrievalWorkspace
            initialDocumentId={targetDocId}
            onSelectDocument={handleNavigateToDoc}
            traceEnabled={traceEnabled}
            onViewTrace={(trace) => {
              setActiveTraceData(trace);
              setActiveTab('trace');
            }}
          />
        )}

        {/* ================= TAB 3: DOCUMENT LIBRARY & VERSIONS ================= */}
        {activeTab === 'documents' && (
          <DocumentWorkspace
            initialDocumentId={targetDocId}
            onNavigateToRetrieval={handleNavigateToRetrieval}
          />
        )}

        {/* ================= TAB 4: RETRIEVAL DEBUG & TRACE ================= */}
        {activeTab === 'trace' && (
          <>
            {activeTraceData || response?.trace ? (
              <TracePanel trace={(activeTraceData || response?.trace)!} />
            ) : (
              <div className="glass-card empty-state" style={{ padding: '3rem' }}>
                <p className="empty-title">No Trace Captured Yet</p>
                <p className="empty-desc">
                  Enable Trace Mode in the header or execute a vector retrieval query to inspect candidates, cosine distances, cross-encoder scores, and context assembly.
                </p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
};
