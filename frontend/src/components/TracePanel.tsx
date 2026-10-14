/**
 * Full-detail, read-only rendering of one `TraceData` payload -- shared by
 * the Research & Synthesis page (inline, when trace is available) and the
 * Retrieval Explorer's "View Trace & Rerank Scores" button. Renders exactly
 * what the backend computed (decomposition, candidate pool, final reranked
 * results, assembled context) and performs none of that logic itself.
 *
 * `trace.candidates` is the deduplicated pool BEFORE reranking (no
 * `rerank_score`); `trace.final_results` is AFTER reranking (has
 * `rerank_score`) -- this mirrors RetrievalService's own distinction, not a
 * display quirk.
 */
import React, { useState } from 'react';
import { TraceData, CandidateResult } from '../types/research';
import { Terminal, ChevronDown, ChevronUp, Layers, CheckCircle, AlignLeft, Sparkles, Filter, GitBranch } from 'lucide-react';

interface TracePanelProps {
  trace: TraceData;
}

export const TracePanel: React.FC<TracePanelProps> = ({ trace }) => {
  const [openCandidateId, setOpenCandidateId] = useState<string | null>(null);
  const [openFinalId, setOpenFinalId] = useState<string | null>(null);
  const [showRawContext, setShowRawContext] = useState(false);

  const toggleCandidate = (id: string) => {
    setOpenCandidateId(openCandidateId === id ? null : id);
  };

  const toggleFinal = (id: string) => {
    setOpenFinalId(openFinalId === id ? null : id);
  };

  return (
    <div className="glass-card trace-panel" style={{ padding: '1.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Terminal size={20} color="#6366f1" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.01em' }}>
            Retrieval & Execution Trace
          </h2>
          <span style={{ fontSize: '0.72rem', background: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc', padding: '0.2rem 0.5rem', borderRadius: 4, fontFamily: 'var(--font-mono)' }}>
            Developer Mode
          </span>
        </div>
      </div>

      {/* Query Decomposition */}
      <div className="trace-section" style={{ marginTop: 0 }}>
        <div className="trace-section-title">
          <GitBranch size={14} color="#818cf8" />
          Query Decomposition
        </div>

        <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.6rem' }}>
          Original query: <span style={{ color: '#e2e8f0', fontWeight: 500 }}>"{trace.original_query}"</span>
        </div>

        {/* The router only ever decides WHETHER the decomposer runs; it
            never generates sub-queries itself. These three states are
            mutually exclusive and derived only from the trace's own
            `router_bypassed`/`sub_queries` fields -- never inferred from
            query text or length. */}
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
            color: trace.router_bypassed ? '#34d399' : '#94a3b8',
            background: trace.router_bypassed
              ? 'rgba(16, 185, 129, 0.12)'
              : 'rgba(148, 163, 184, 0.1)',
            border: `1px solid ${trace.router_bypassed ? 'rgba(16, 185, 129, 0.3)' : 'rgba(148, 163, 184, 0.2)'}`,
          }}
        >
          <GitBranch size={11} />
          {trace.router_bypassed
            ? 'Query Router: bypassed decomposition (single-intent query)'
            : 'Query Router: allowed decomposition'}
        </div>

        {trace.router_bypassed ? (
          <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
            The router determined this query was confidently single-intent and skipped
            the decomposer entirely -- the original query above was used directly for
            retrieval.
          </div>
        ) : trace.sub_queries.length > 1 ? (
          <>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
              The decomposer split this query into {trace.sub_queries.length} retrieval
              sub-queries, each run independently and merged into one candidate pool:
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              {trace.sub_queries.map((subQuery, idx) => (
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
          <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
            The decomposer ran (the router did not bypass it) and determined a single
            information need, so retrieval ran against the original query only.
          </div>
        )}
      </div>

      {/* Summary Metrics Grid */}
      <div className="trace-summary-grid">
        <div className="trace-metric-card">
          <div className="trace-metric-label">Candidate Limit</div>
          <div className="trace-metric-value" style={{ color: '#06b6d4' }}>
            {trace.candidate_limit}
          </div>
        </div>

        <div className="trace-metric-card">
          <div className="trace-metric-label">Raw Candidates</div>
          <div className="trace-metric-value" style={{ color: '#818cf8' }}>
            {trace.raw_candidate_count}
          </div>
        </div>

        <div className="trace-metric-card">
          <div className="trace-metric-label">Deduplicated Candidates</div>
          <div className="trace-metric-value" style={{ color: '#a855f7' }}>
            {trace.deduplicated_candidate_count}
          </div>
        </div>

        <div className="trace-metric-card">
          <div className="trace-metric-label">Final Results Selected</div>
          <div className="trace-metric-value" style={{ color: '#10b981' }}>
            {trace.final_results?.length || 0}
          </div>
        </div>
      </div>

      {/* 1. CANDIDATE RETRIEVAL SECTION */}
      <div className="trace-section">
        <div className="trace-section-title">
          <Layers size={14} color="#06b6d4" />
          1. Candidate Retrieval ({trace.candidates?.length || 0} candidates)
        </div>
        <div>
          {trace.candidates && trace.candidates.length > 0 ? (
            trace.candidates.map((cand: CandidateResult, idx: number) => {
              const isOpen = openCandidateId === `cand-${idx}-${cand.chunk_id}`;
              return (
                <div key={`cand-${idx}-${cand.chunk_id}`} className="trace-chunk-accordion">
                  <div
                    className="trace-chunk-header"
                    onClick={() => toggleCandidate(`cand-${idx}-${cand.chunk_id}`)}
                  >
                    <div className="trace-chunk-title">
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#64748b', fontSize: '0.8rem' }}>
                        #{idx + 1}
                      </span>
                      <span>{cand.section_path || 'Section Content'}</span>
                    </div>
                    <div className="trace-scores-row">
                      <span className="score-tag-distance">
                        Distance: {cand.distance?.toFixed(4) ?? 'N/A'}
                      </span>
                      {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>
                  {isOpen && (
                    <div className="trace-chunk-content">
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '0.75rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                        <span style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
                          Doc ID: <strong style={{ color: '#f8fafc' }}>{cand.document_id}</strong>
                        </span>
                        <span style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
                          Chunk ID: <strong style={{ color: '#f8fafc' }}>{cand.chunk_id}</strong>
                        </span>
                        <span style={{ minWidth: 0, overflowWrap: 'anywhere' }}>
                          Pages: <strong style={{ color: '#f8fafc' }}>{cand.page_numbers?.join(', ') || 'N/A'}</strong>
                        </span>
                      </div>
                      <div>{cand.content}</div>
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <p style={{ fontSize: '0.85rem', color: '#64748b' }}>No retrieved candidates recorded in trace.</p>
          )}
        </div>
      </div>

      {/* 2. RERANKING & FINAL RESULTS SECTION */}
      <div className="trace-section">
        <div className="trace-section-title">
          <Filter size={14} color="#10b981" />
          2. Cross-Encoder Reranking & Final Selection ({trace.final_results?.length || 0} selected)
        </div>
        <div>
          {trace.final_results && trace.final_results.length > 0 ? (
            trace.final_results.map((res: CandidateResult, idx: number) => {
              const isOpen = openFinalId === `final-${idx}-${res.chunk_id}`;
              return (
                <div key={`final-${idx}-${res.chunk_id}`} className="trace-chunk-accordion" style={{ borderColor: 'rgba(16, 185, 129, 0.3)' }}>
                  <div
                    className="trace-chunk-header"
                    onClick={() => toggleFinal(`final-${idx}-${res.chunk_id}`)}
                  >
                    <div className="trace-chunk-title">
                      <CheckCircle size={14} color="#10b981" />
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#64748b', fontSize: '0.8rem' }}>
                        Selected #{idx + 1}
                      </span>
                      <span>{res.section_path || 'Section Content'}</span>
                    </div>
                    <div className="trace-scores-row">
                      <span className="score-tag-distance">
                        Distance: {res.distance?.toFixed(4) ?? 'N/A'}
                      </span>
                      <span className="score-tag-rerank">
                        Rerank Score: {res.rerank_score?.toFixed(3) ?? 'N/A'}
                      </span>
                      {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </div>
                  </div>
                  {isOpen && (
                    <div className="trace-chunk-content">
                      <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '0.75rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                        <span>Doc ID: <strong style={{ color: '#f8fafc' }}>{res.document_id}</strong></span>
                        <span>Chunk ID: <strong style={{ color: '#f8fafc' }}>{res.chunk_id}</strong></span>
                        <span>Page(s): <strong style={{ color: '#f8fafc' }}>{res.page_numbers?.join(', ') || 'N/A'}</strong></span>
                      </div>
                      <div>{res.content}</div>
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <p style={{ fontSize: '0.85rem', color: '#64748b' }}>No final results selected after reranking.</p>
          )}
        </div>
      </div>

      {/* 3. CONTEXT SENT TO GENERATION */}
      {trace.context && (
        <div className="trace-section">
          <div className="trace-section-title" style={{ justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlignLeft size={14} color="#818cf8" />
              3. Context Sent to Generation
            </span>
            <button
              type="button"
              className="btn-secondary"
              style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem' }}
              onClick={() => setShowRawContext(!showRawContext)}
            >
              {showRawContext ? 'Hide Raw Context' : 'View Assembled Text'}
            </button>
          </div>

          {showRawContext ? (
            <div className="context-box">
              {trace.context.text}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {trace.context.sources?.map((src, idx) => (
                <div key={idx} className="trace-chunk-accordion" style={{ padding: '0.85rem 1.2rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontWeight: 600, fontSize: '0.85rem', color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Sparkles size={12} /> [Source {idx + 1}] {src.section_path || 'Section Content'}
                    </span>
                    <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                      Page {src.page_numbers?.join(', ')}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#cbd5e1', fontFamily: 'var(--font-mono)', lineHeight: 1.5, background: 'rgba(0,0,0,0.3)', padding: '0.6rem 0.8rem', borderRadius: 6 }}>
                    {src.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Observability & Langfuse note */}
      <div
        style={{
          marginTop: '1.5rem',
          padding: '0.85rem 1.25rem',
          background: 'rgba(15, 23, 42, 0.5)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 8,
          fontSize: '0.75rem',
          color: '#94a3b8',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span>
          <strong>Observability Integration:</strong> Backend traces each retrieval step (`retriever` span) and generation request to Langfuse when `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` are configured.
        </span>
        <span style={{ color: '#818cf8', fontWeight: 600 }}>External Langfuse Dashboard</span>
      </div>
    </div>
  );
};
