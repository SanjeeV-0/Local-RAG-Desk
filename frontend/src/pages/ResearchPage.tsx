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
import { useCancellableRequest } from '../hooks/useCancellableRequest';
import { DecompositionOutcomeSummary } from '../components/DecompositionOutcomeSummary';
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
import { AlertTriangle, RefreshCw, XCircle } from 'lucide-react';

export const ResearchPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('research');
  const [traceEnabled, setTraceEnabled] = useState<boolean>(false);
  // Global Query Router toggle -- a per-request override, defaults OFF to
  // match the server's own default (`Settings.query_router_enabled`).
  // Deliberately not persisted across reloads: no existing preference
  // mechanism exists in this app (Trace Mode isn't persisted either), and
  // the task calls for a fresh default each load unless one already did.
  const [routerEnabled, setRouterEnabled] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<ResearchResponse | null>(null);
  const [activeTraceData, setActiveTraceData] = useState<TraceData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCancelled, setIsCancelled] = useState<boolean>(false);
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [readinessInfo, setReadinessInfo] = useState<BackendReadinessResponse | null>(null);
  const [lastQuery, setLastQuery] = useState<string>('');

  const researchRequest = useCancellableRequest();

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
    const { signal, isCurrent } = researchRequest.start();

    setIsLoading(true);
    setError(null);
    setIsCancelled(false);
    setLastQuery(query);

    try {
      const data = await askResearchQuestion(query, traceEnabled, routerEnabled, signal);

      // A newer request may have already superseded this one (the user
      // submitted again, or cancelled-and-resubmitted) -- if so, this
      // result is stale and must never overwrite what the newer request
      // already established.
      if (!isCurrent()) return;

      setResponse(data);
      setIsBackendHealthy(true);

      if (data.trace) {
        setActiveTraceData(data.trace);
      }
    } catch (err: any) {
      // Superseded by a newer request -- that request owns the UI now;
      // this one has nothing left to say, whether it failed or was
      // itself aborted as a side effect of being superseded.
      if (!isCurrent()) return;

      if (err?.name === 'AbortError') {
        // A genuine user-initiated cancellation of what is still the
        // current request (the branch above already filtered out the
        // "aborted because superseded" case). This is NOT a backend/
        // network failure -- do not show the generic error banner, and
        // do not claim the backend actually stopped processing; the
        // fetch was simply abandoned client-side.
        setIsCancelled(true);
        setResponse(null);
        return;
      }

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
      // A superseded request's `finally` must never clear the newer
      // request's loading state.
      if (isCurrent()) setIsLoading(false);
    }
  };

  const handleCancelResearch = () => {
    researchRequest.cancel();
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
        routerEnabled={routerEnabled}
        onToggleRouter={setRouterEnabled}
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
              onCancel={handleCancelResearch}
            />

            {isCancelled && (
              <div className="warning-banner">
                <XCircle size={22} style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <div className="warning-title">Request Cancelled</div>
                  <div className="warning-desc">
                    The request was cancelled before a response was received. This only
                    stops the browser from waiting on or using the response -- it does
                    not guarantee any in-progress backend processing was halted.
                  </div>
                </div>
              </div>
            )}

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

                {/* Always-visible query decomposition outcome -- present on
                    every completed response regardless of Trace Mode. Only
                    rendered when the fuller TracePanel below is NOT shown,
                    so the two never appear at once. */}
                {response?.decomposition && !(traceEnabled && response?.trace) && (
                  <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
                    <DecompositionOutcomeSummary outcome={response.decomposition} />
                  </div>
                )}

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
            routerEnabled={routerEnabled}
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
