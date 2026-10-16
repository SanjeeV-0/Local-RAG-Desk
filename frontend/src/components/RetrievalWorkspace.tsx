/**
 * Retrieval Explorer -- the engineering/debugging workbench for retrieval,
 * deliberately NOT a second RAG answer UI (it only calls
 * POST /retrieval/search, never /research/ask, so it never triggers
 * generation). Its purpose is "show me why these chunks were retrieved":
 * the original query, whether/how it was decomposed, raw vs. deduplicated
 * candidate counts, and the final reranked results with full provenance.
 *
 * UI state is driven by the response's explicit `trace_requested` /
 * `trace_available` / `trace_unavailable_reason` fields, never by
 * `response.trace`'s truthiness alone -- this is what lets the component
 * tell "Trace Mode was off" apart from "Trace Mode was on but the server
 * has tracing disabled" and show the right message for each.
 */
import React, { useEffect, useMemo, useState } from 'react';
import {
  RetrievalSearchRequest,
  RetrievalSearchResponse,
  RetrievedChunkResponse,
  TraceData,
} from '../types/research';
import { searchRetrievedChunks } from '../api/research';
import { listDocuments, DocumentApiUnavailableError } from '../api/documents';
import { LogicalDocumentSummary } from '../types/document';
import { useCancellableRequest } from '../hooks/useCancellableRequest';
import { DecompositionOutcomeSummary } from './DecompositionOutcomeSummary';
import { RedactionWarningBanner } from './RedactionWarningBanner';
import {
  Search,
  Layers,
  BookOpen,
  Hash,
  FileText,
  Sliders,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Copy,
  Check,
  Cpu,
  GitBranch,
  Terminal,
  X,
  XCircle,
} from 'lucide-react';

interface RetrievalWorkspaceProps {
  initialDocumentId?: string | null;
  onSelectDocument?: (documentId: string) => void;
  onViewTrace?: (trace: TraceData) => void;
  // The header's global Trace Mode toggle. This is the single source of
  // truth for whether this page's requests ask the backend for a trace --
  // there is no separate, locally-duplicated toggle here, so the UI state
  // and the request payload can never drift apart.
  traceEnabled: boolean;
  // The header's global Query Router toggle -- same single-source-of-truth
  // rationale as `traceEnabled`, sent explicitly as `router_enabled` on
  // every request.
  routerEnabled: boolean;
}

export const RetrievalWorkspace: React.FC<RetrievalWorkspaceProps> = ({
  initialDocumentId,
  onSelectDocument,
  onViewTrace,
  traceEnabled,
  routerEnabled,
}) => {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(10);
  const [candidateLimitInput, setCandidateLimitInput] = useState('');
  const [selectedDocumentId, setSelectedDocumentId] = useState(initialDocumentId || '');
  const [documentFilterText, setDocumentFilterText] = useState('');
  const [sectionIdFilter, setSectionIdFilter] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(!!initialDocumentId);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCancelled, setIsCancelled] = useState(false);
  const [response, setResponse] = useState<RetrievalSearchResponse | null>(null);
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Document selector state -- loaded once via the existing GET /documents
  // wrapper (`listDocuments`), never re-fetched per keystroke. The filter
  // text below only narrows this already-loaded list client-side.
  const [documents, setDocuments] = useState<LogicalDocumentSummary[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [documentsApiUnavailable, setDocumentsApiUnavailable] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const loadDocuments = async () => {
      setDocumentsLoading(true);
      setDocumentsError(null);

      try {
        const docs = await listDocuments();
        if (cancelled) return;
        setDocuments(docs);
        setDocumentsApiUnavailable(false);
      } catch (err: any) {
        if (cancelled) return;
        if (err instanceof DocumentApiUnavailableError) {
          setDocumentsApiUnavailable(true);
          setDocuments([]);
        } else {
          setDocumentsError(err?.message || 'Failed to load documents.');
        }
      } finally {
        if (!cancelled) setDocumentsLoading(false);
      }
    };

    loadDocuments();

    return () => {
      cancelled = true;
    };
  }, []);

  // Builds a label that stays useful even when several documents share the
  // same title -- version number and non-READY status are only appended
  // when they actually help disambiguate, so the common case (one READY
  // version, unique title) stays a plain, uncluttered title.
  //
  // `id` is the CURRENT VERSION's id (`doc.current_version.id`), NOT
  // `doc.logical_document_id` -- verified against the backend contract:
  // `POST /retrieval/search`'s `document_id` filters
  // `DocumentChunk.document_id`, a foreign key to the per-version
  // `Document.id` (see `DocumentRepository.search_similar_chunks` and its
  // tests), never the logical grouping id. Sending `logical_document_id`
  // here -- a value that never equals any chunk's `document_id` -- was why
  // selecting a specific document always returned zero results, while the
  // unfiltered (no document_id) search worked fine. A document with no
  // current version has nothing to filter to, so it is kept visible (for
  // information) but not selectable.
  const documentOptions = useMemo(() => {
    const titleCounts = new Map<string, number>();
    for (const doc of documents) {
      const title = doc.current_version?.title ?? 'Untitled document';
      titleCounts.set(title, (titleCounts.get(title) ?? 0) + 1);
    }

    return documents.map((doc) => {
      const title = doc.current_version?.title ?? 'Untitled document';
      const isDuplicateTitle = (titleCounts.get(title) ?? 0) > 1;

      let label = title;
      if (!doc.current_version) {
        label += ` (no current version -- unavailable for filtering)`;
      } else {
        if (isDuplicateTitle) {
          label += ` (v${doc.current_version.version_number})`;
        }
        if (doc.current_version.status !== 'READY') {
          label += ` [${doc.current_version.status}]`;
        }
      }

      return {
        key: doc.logical_document_id,
        id: doc.current_version?.id ?? '',
        label,
        selectable: !!doc.current_version,
        searchText: `${title} ${doc.logical_document_id}`.toLowerCase(),
      };
    });
  }, [documents]);

  const filteredDocumentOptions = useMemo(() => {
    const needle = documentFilterText.trim().toLowerCase();
    if (!needle) return documentOptions;
    return documentOptions.filter((opt) => opt.searchText.includes(needle));
  }, [documentOptions, documentFilterText]);

  const retrievalRequest = useCancellableRequest();

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;

    const { signal, isCurrent } = retrievalRequest.start();

    setIsLoading(true);
    setError(null);
    setIsCancelled(false);

    const payload: RetrievalSearchRequest = {
      query: query.trim(),
      limit,
      trace: traceEnabled,
      router_enabled: routerEnabled,
    };
    if (selectedDocumentId) {
      payload.document_id = selectedDocumentId;
    }
    if (sectionIdFilter.trim()) {
      payload.section_id = sectionIdFilter.trim();
    }
    const parsedCandidateLimit = candidateLimitInput.trim()
      ? Number(candidateLimitInput)
      : undefined;
    if (parsedCandidateLimit !== undefined && Number.isFinite(parsedCandidateLimit)) {
      payload.candidate_limit = parsedCandidateLimit;
    }

    try {
      const data = await searchRetrievedChunks(payload, signal);

      // Superseded by a newer request -- never overwrite its state.
      if (!isCurrent()) return;

      setResponse(data);

      if (data.trace_available && data.trace && onViewTrace) {
        onViewTrace(data.trace);
      }
    } catch (err: any) {
      if (!isCurrent()) return;

      if (err?.name === 'AbortError') {
        // Genuine user cancellation of the still-current request -- not a
        // backend/network failure, so no generic error banner. Aborting
        // the fetch only stops the browser from waiting on the response;
        // it does not guarantee the backend stopped processing.
        setIsCancelled(true);
        setResponse(null);
        return;
      }

      setError(err?.message || 'Failed to execute vector retrieval search.');
      // Clear the previous successful response so stale chunks from an
      // earlier query can never be displayed as though they answer this
      // failed request.
      setResponse(null);
    } finally {
      if (isCurrent()) setIsLoading(false);
    }
  };

  const handleCancelSearch = () => {
    retrievalRequest.cancel();
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="retrieval-workspace">
      {/* Parameter Controls Panel */}
      <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <form onSubmit={handleSearch}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Search size={18} color="#06b6d4" />
              <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                Vector Retrieval Explorer
              </h2>
            </div>
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{ fontSize: '0.78rem' }}
            >
              <Sliders size={14} />
              {showAdvanced ? 'Hide Retrieval Filters' : 'Retrieval Filters'}
            </button>
          </div>

          <div className="input-wrapper" style={{ marginBottom: '1rem' }}>
            <textarea
              className="research-textarea"
              rows={2}
              placeholder="Enter search query or concept to embed with SentenceTransformer and rank via pgvector..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isLoading}
            />
          </div>

          {showAdvanced && (
            <div className="advanced-filters-grid" style={{ marginBottom: '1.25rem' }}>
              <div>
                <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                  Final Result Limit (Top {limit} chunks returned)
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <input
                    type="range"
                    min={1}
                    max={50}
                    value={limit}
                    onChange={(e) => setLimit(Number(e.target.value))}
                    style={{ flex: 1, accentColor: '#06b6d4' }}
                    disabled={isLoading}
                  />
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={limit}
                    onChange={(e) => setLimit(Math.max(1, Math.min(50, Number(e.target.value))))}
                    style={{
                      width: 60,
                      background: 'rgba(15,23,42,0.8)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 6,
                      color: '#f8fafc',
                      padding: '0.2rem 0.5rem',
                      fontSize: '0.8rem',
                      fontFamily: 'var(--font-mono)',
                    }}
                    disabled={isLoading}
                  />
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.3rem' }}>
                  The number of reranked results returned -- not the number of candidates
                  evaluated before reranking (see "Candidate Limit per Subquery" below).
                </div>
              </div>

              <div>
                <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                  Candidate Limit per Subquery (optional)
                </label>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  placeholder="Server default applies"
                  value={candidateLimitInput}
                  onChange={(e) => setCandidateLimitInput(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 6,
                    color: '#f8fafc',
                    padding: '0.45rem 0.75rem',
                    fontSize: '0.8rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                  disabled={isLoading}
                />
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.3rem' }}>
                  Candidates are fetched per subquery, then merged and deduplicated -- the
                  combined pool can exceed this number when the query is decomposed into
                  more than one subquery. Leave blank to use the server's configured
                  default (shown in the trace summary below as "Candidate Limit" once a
                  request completes with Trace Mode on).
                </div>
              </div>

              <div>
                <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                  Document Filter
                </label>
                <input
                  type="text"
                  placeholder="Filter documents by title..."
                  value={documentFilterText}
                  onChange={(e) => setDocumentFilterText(e.target.value)}
                  style={{
                    width: '100%',
                    marginBottom: '0.4rem',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 6,
                    color: '#f8fafc',
                    padding: '0.45rem 0.75rem',
                    fontSize: '0.8rem',
                  }}
                  disabled={isLoading || documentsLoading || documentsApiUnavailable}
                />
                <select
                  value={selectedDocumentId}
                  onChange={(e) => setSelectedDocumentId(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 6,
                    color: '#f8fafc',
                    padding: '0.45rem 0.75rem',
                    fontSize: '0.8rem',
                  }}
                  disabled={isLoading || documentsLoading || documentsApiUnavailable}
                >
                  <option value="">— All documents (no filter) —</option>
                  {filteredDocumentOptions.map((opt) => (
                    <option key={opt.key} value={opt.id} disabled={!opt.selectable}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.3rem' }}>
                  {documentsLoading
                    ? 'Loading documents...'
                    : documentsApiUnavailable
                    ? 'Document list API is not available -- document filtering is unavailable.'
                    : documentsError
                    ? `Failed to load documents: ${documentsError}`
                    : documents.length === 0
                    ? 'No documents found.'
                    : filteredDocumentOptions.length === 0 && documentFilterText.trim()
                    ? 'No documents match that filter.'
                    : 'Select "All documents" to clear the filter and search the whole corpus.'}
                </div>
              </div>

              <div>
                <label className="input-label" style={{ fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                  Section Filter (UUID)
                </label>
                <input
                  type="text"
                  placeholder="Paste a section UUID (optional)"
                  value={sectionIdFilter}
                  onChange={(e) => setSectionIdFilter(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 6,
                    color: '#f8fafc',
                    padding: '0.45rem 0.75rem',
                    fontSize: '0.8rem',
                    fontFamily: 'var(--font-mono)',
                  }}
                  disabled={isLoading}
                />
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.3rem' }}>
                  No section browser is available yet -- paste a section UUID directly
                  (e.g. copied from a result's "Details" drawer below).
                </div>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.75rem' }}>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.35rem',
                fontSize: '0.72rem',
                fontWeight: 600,
                color: traceEnabled ? '#818cf8' : '#64748b',
                marginRight: 'auto',
              }}
              title={
                traceEnabled
                  ? 'Trace Mode is ON: this request asks the backend to capture a full retrieval trace (subject to server-side TRACE_ENABLED).'
                  : 'Trace Mode is OFF: only final retrieved chunks will be returned, no trace.'
              }
            >
              <Terminal size={12} />
              Trace Mode {traceEnabled ? 'ON' : 'OFF'} for this request
            </span>
            {isLoading && (
              <button type="button" className="btn-secondary" onClick={handleCancelSearch}>
                <X size={14} />
                Cancel
              </button>
            )}
            <button
              type="submit"
              className="btn-primary"
              disabled={!query.trim() || isLoading}
              style={{ background: 'linear-gradient(135deg, #06b6d4, #3b82f6)' }}
            >
              {isLoading ? (
                <>
                  <span className="spinner-ring" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  Searching pgvector...
                </>
              ) : (
                <>
                  <Search size={16} />
                  Execute Vector Retrieval
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Cancelled state -- a user-initiated abort, not a backend/network
          failure, so it is never shown alongside (or worded like) the
          error banner below. */}
      {isCancelled && (
        <div className="warning-banner">
          <XCircle size={22} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div className="warning-title">Request Cancelled</div>
            <div className="warning-desc">
              The request was cancelled before a response was received. This only stops
              the browser from waiting on or using the response -- it does not guarantee
              any in-progress backend processing was halted.
            </div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="error-banner">
          <AlertCircle size={22} style={{ flexShrink: 0, marginTop: 2, color: '#f43f5e' }} />
          <div>
            <h4 className="error-title">Retrieval API Error</h4>
            <p className="error-desc">{error}</p>
          </div>
        </div>
      )}

      {response?.redaction?.possibly_incomplete && (
        <div style={{ marginBottom: '1.5rem' }}>
          <RedactionWarningBanner redaction={response.redaction} />
        </div>
      )}

      {/* Always-visible query decomposition outcome -- present on every
          completed response regardless of Trace Mode. Only rendered when
          the fuller, trace-gated "Query Decomposition" card below is NOT
          available, so the two are never shown at once; when tracing is
          available, that richer card (candidate/dedup counts included)
          takes over. */}
      {response?.decomposition && !response.trace_available && (
        <div className="glass-card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.5rem' }}>
          <DecompositionOutcomeSummary outcome={response.decomposition} />
        </div>
      )}

      {/* Results Header / Stats */}
      {response && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', padding: '0 0.5rem' }}>
          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Found <strong style={{ color: '#f8fafc' }}>{response.results.length}</strong> matching chunks
            {response.trace_available && response.trace && (
              <span style={{ marginLeft: '0.75rem', color: '#06b6d4' }}>
                (from {response.trace.candidate_limit} pgvector candidates)
              </span>
            )}
          </div>
          {response.trace_available && response.trace && (
            <button
              type="button"
              className="btn-secondary"
              style={{ fontSize: '0.75rem', padding: '0.3rem 0.75rem' }}
              onClick={() => onViewTrace && onViewTrace(response.trace!)}
            >
              <Cpu size={13} /> View Trace & Rerank Scores
            </button>
          )}
        </div>
      )}

      {/* Trace Mode was ON for this request, but the server has tracing
          disabled. Retrieval and query decomposition still ran normally --
          only the trace representation of that execution is unavailable.
          This must never be shown just because Trace Mode was OFF. */}
      {response?.trace_requested && !response.trace_available && response.trace_unavailable_reason === 'server_disabled' && (
        <div className="warning-banner" style={{ marginBottom: '1.5rem' }}>
          <AlertCircle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div className="warning-title">Trace Unavailable</div>
            <div className="warning-desc">
              Trace Mode was requested, but backend trace capture is disabled
              (<code>TRACE_ENABLED=false</code>). Retrieval is still running normally
              (query decomposition may or may not run, depending on the Query Router
              setting). Only the final retrieved chunks are shown -- the router's
              decision and any sub-queries are part of the trace, so they are not
              available here either.
            </div>
          </div>
        </div>
      )}

      {/* Why these chunks: query decomposition + candidate pool summary.
          This is the Retrieval Explorer's core purpose, so it is shown
          inline rather than only behind the separate trace view. */}
      {response?.trace_available && response.trace && (
        <div className="glass-card" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.85rem' }}>
            <GitBranch size={15} color="#818cf8" />
            <h3 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#f8fafc' }}>
              Query Decomposition
            </h3>
          </div>

          <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
            Original query: <span style={{ color: '#e2e8f0', fontWeight: 500 }}>"{response.trace.original_query}"</span>
          </div>

          {/* The router only ever decides WHETHER the decomposer runs; it
              never generates sub-queries itself -- that distinction is kept
              explicit in the three mutually-exclusive states below, derived
              only from the trace's own `router_bypassed`/`sub_queries`
              fields, never inferred from query text or length. */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontSize: '0.72rem',
              fontWeight: 600,
              padding: '0.2rem 0.6rem',
              borderRadius: 9999,
              marginBottom: '0.75rem',
              color: response.trace.router_bypassed ? '#34d399' : '#94a3b8',
              background: response.trace.router_bypassed
                ? 'rgba(16, 185, 129, 0.12)'
                : 'rgba(148, 163, 184, 0.1)',
              border: `1px solid ${response.trace.router_bypassed ? 'rgba(16, 185, 129, 0.3)' : 'rgba(148, 163, 184, 0.2)'}`,
            }}
          >
            <GitBranch size={11} />
            {response.trace.router_bypassed
              ? 'Query Router: bypassed decomposition (single-intent query)'
              : 'Query Router: allowed decomposition'}
          </div>

          {response.trace.router_bypassed ? (
            <div style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '1rem' }}>
              The router determined this query was confidently single-intent and skipped
              the decomposer entirely -- the original query above was used directly for
              retrieval.
            </div>
          ) : response.trace.sub_queries.length > 1 ? (
            <>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
                The decomposer split this query into{' '}
                <strong style={{ color: '#818cf8' }}>{response.trace.sub_queries.length}</strong>{' '}
                retrieval sub-queries, each run independently and merged:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '1rem' }}>
                {response.trace.sub_queries.map((subQuery, idx) => (
                  <span
                    key={idx}
                    style={{
                      fontSize: '0.75rem',
                      color: '#a5b4fc',
                      background: 'rgba(99, 102, 241, 0.12)',
                      border: '1px solid rgba(99, 102, 241, 0.25)',
                      padding: '0.25rem 0.6rem',
                      borderRadius: 9999,
                    }}
                  >
                    {subQuery}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div style={{ fontSize: '0.78rem', color: '#64748b', marginBottom: '1rem' }}>
              The decomposer ran (the router did not bypass it) and determined a single
              information need, so retrieval ran against the original query only.
            </div>
          )}

          <div className="advanced-filters-grid" style={{ marginBottom: 0 }}>
            <div className="trace-metric-card" style={{ padding: '0.75rem 1rem' }}>
              <div className="trace-metric-label">Candidate Limit</div>
              <div className="trace-metric-value" style={{ fontSize: '1.05rem', color: '#06b6d4' }}>
                {response.trace.candidate_limit}
              </div>
            </div>
            <div className="trace-metric-card" style={{ padding: '0.75rem 1rem' }}>
              <div className="trace-metric-label">Raw Candidates</div>
              <div className="trace-metric-value" style={{ fontSize: '1.05rem', color: '#818cf8' }}>
                {response.trace.raw_candidate_count}
              </div>
            </div>
            <div className="trace-metric-card" style={{ padding: '0.75rem 1rem' }}>
              <div className="trace-metric-label">Deduplicated Candidates</div>
              <div className="trace-metric-value" style={{ fontSize: '1.05rem', color: '#a855f7' }}>
                {response.trace.deduplicated_candidate_count}
              </div>
            </div>
            <div className="trace-metric-card" style={{ padding: '0.75rem 1rem' }}>
              <div className="trace-metric-label">Final Reranked Top K</div>
              <div className="trace-metric-value" style={{ fontSize: '1.05rem', color: '#10b981' }}>
                {response.trace.final_results.length}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {isLoading ? (
          <div className="glass-card" style={{ padding: '3rem', textAlign: 'center' }}>
            <div className="spinner-outer" style={{ width: 40, height: 40, margin: '0 auto 1rem auto' }}>
              <div className="spinner-ring" style={{ borderTopColor: '#06b6d4' }} />
            </div>
            <p style={{ color: '#f8fafc', fontWeight: 600 }}>Querying pgvector vector store...</p>
            <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '0.25rem' }}>
              Filtering to current READY versions and applying cosine similarity.
            </p>
          </div>
        ) : response && response.results.length > 0 ? (
          response.results.map((chunk: RetrievedChunkResponse, idx: number) => {
            const isExpanded = expandedChunkId === chunk.chunk_id;
            const pagesText = chunk.page_numbers?.length ? chunk.page_numbers.join(', ') : 'N/A';
            const similarityPercent = (chunk.similarity * 100).toFixed(1);

            return (
              <div key={chunk.chunk_id || idx} className="glass-card" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <span className="source-badge-index" style={{ background: 'rgba(6, 182, 212, 0.2)', color: '#22d3ee' }}>
                      #{idx + 1}
                    </span>
                    <div>
                      <div style={{ fontWeight: 600, color: '#f8fafc', fontSize: '0.95rem' }}>
                        {chunk.section_path || 'Document Content'}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.25rem' }}>
                        <span className="source-pages">
                          <BookOpen size={11} /> Page {pagesText}
                        </span>
                        <span style={{ fontSize: '0.72rem', color: '#06b6d4', fontFamily: 'var(--font-mono)' }}>
                          Cosine Dist: {chunk.distance.toFixed(4)}
                        </span>
                        <span style={{ fontSize: '0.72rem', color: '#10b981', fontFamily: 'var(--font-mono)' }}>
                          Similarity: {similarityPercent}%
                        </span>
                        {chunk.rerank_score !== null && chunk.rerank_score !== undefined && (
                          <span style={{ fontSize: '0.72rem', color: '#a855f7', fontFamily: 'var(--font-mono)' }}>
                            Rerank: {chunk.rerank_score.toFixed(3)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {onSelectDocument && (
                      <button
                        type="button"
                        className="btn-ghost"
                        style={{ fontSize: '0.72rem', padding: '0.25rem 0.5rem' }}
                        onClick={() => onSelectDocument(chunk.document_id)}
                        title="View document in Document Workspace"
                      >
                        <FileText size={12} /> Inspect Doc
                      </button>
                    )}
                    <button
                      type="button"
                      className="btn-ghost"
                      style={{ fontSize: '0.72rem', padding: '0.25rem 0.5rem' }}
                      onClick={() => setExpandedChunkId(isExpanded ? null : chunk.chunk_id)}
                    >
                      {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      {isExpanded ? 'Collapse' : 'Details'}
                    </button>
                  </div>
                </div>

                {/* Chunk text content */}
                <div
                  style={{
                    marginTop: '0.85rem',
                    padding: '0.85rem 1rem',
                    background: 'rgba(15, 23, 42, 0.75)',
                    border: '1px solid rgba(255,255,255,0.05)',
                    borderRadius: 8,
                    fontSize: '0.85rem',
                    lineHeight: '1.6',
                    color: '#e2e8f0',
                    fontFamily: 'var(--font-sans)',
                  }}
                >
                  {chunk.content}
                </div>

                {/* Expandable Technical Details Drawer */}
                {isExpanded && (
                  <div className="source-details-drawer" style={{ marginTop: '0.85rem' }}>
                    <div className="source-detail-item">
                      <span className="source-detail-label">
                        <FileText size={10} style={{ display: 'inline', marginRight: 2 }} /> Document Version ID
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span className="source-detail-value" title={chunk.document_id}>
                          {chunk.document_id}
                        </span>
                        <button
                          type="button"
                          className="btn-ghost"
                          style={{ padding: '0.1rem 0.2rem' }}
                          onClick={() => handleCopy(chunk.document_id, `doc-${chunk.chunk_id}`)}
                        >
                          {copiedId === `doc-${chunk.chunk_id}` ? <Check size={11} color="#10b981" /> : <Copy size={11} />}
                        </button>
                      </div>
                    </div>

                    <div className="source-detail-item">
                      <span className="source-detail-label">
                        <Hash size={10} style={{ display: 'inline', marginRight: 2 }} /> Chunk ID
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span className="source-detail-value" title={chunk.chunk_id}>
                          {chunk.chunk_id}
                        </span>
                        <button
                          type="button"
                          className="btn-ghost"
                          style={{ padding: '0.1rem 0.2rem' }}
                          onClick={() => handleCopy(chunk.chunk_id, `chunk-${chunk.chunk_id}`)}
                        >
                          {copiedId === `chunk-${chunk.chunk_id}` ? <Check size={11} color="#10b981" /> : <Copy size={11} />}
                        </button>
                      </div>
                    </div>

                    <div className="source-detail-item">
                      <span className="source-detail-label">
                        <Layers size={10} style={{ display: 'inline', marginRight: 2 }} /> Section ID
                      </span>
                      <span className="source-detail-value" title={chunk.section_id ?? undefined}>
                        {chunk.section_id || 'N/A'}
                      </span>
                    </div>

                    <div className="source-detail-item">
                      <span className="source-detail-label">
                        <BookOpen size={10} style={{ display: 'inline', marginRight: 2 }} /> Page Numbers
                      </span>
                      <span className="source-detail-value">
                        {pagesText}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        ) : response && response.results.length === 0 ? (
          <div className="glass-card empty-state">
            <AlertCircle className="empty-icon" />
            <p className="empty-title">No Chunks Matched</p>
            <p className="empty-desc">
              No READY current document chunks were close enough in vector space. Try adjusting the query or relaxing filter criteria.
            </p>
          </div>
        ) : (
          <div className="glass-card empty-state">
            <Search className="empty-icon" color="#06b6d4" />
            <p className="empty-title">Vector Retrieval Ready</p>
            <p className="empty-desc">
              Enter a research concept or query above to run vector similarity retrieval directly against indexed chunks.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
