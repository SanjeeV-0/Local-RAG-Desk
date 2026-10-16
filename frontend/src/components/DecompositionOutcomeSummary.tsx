/**
 * Compact, always-available rendering of a `DecompositionOutcome` -- shared
 * by the Research tab and the Retrieval Explorer so the Query Router's
 * decision and the decomposer's actual sub-queries are visible on every
 * completed request, independent of Trace Mode.
 *
 * Deliberately NOT a replacement for the existing, richer trace-gated
 * "Query Decomposition" views (RetrievalWorkspace's inline card,
 * TracePanel) -- those still show candidate/dedup counts and remain
 * Trace-Mode-only. Both call sites render THIS component only when the
 * fuller trace view is unavailable, so the two never show side by side.
 */
import React from 'react';
import { GitBranch } from 'lucide-react';
import { DecompositionOutcome } from '../types/research';

interface DecompositionOutcomeSummaryProps {
  outcome: DecompositionOutcome;
}

const TONE_COLORS = {
  bypassed: {
    color: '#34d399',
    background: 'rgba(16, 185, 129, 0.12)',
    border: 'rgba(16, 185, 129, 0.3)',
  },
  neutral: {
    color: '#94a3b8',
    background: 'rgba(148, 163, 184, 0.1)',
    border: 'rgba(148, 163, 184, 0.2)',
  },
  fallback: {
    color: '#fbbf24',
    background: 'rgba(245, 158, 11, 0.12)',
    border: 'rgba(245, 158, 11, 0.3)',
  },
} as const;

export const DecompositionOutcomeSummary: React.FC<DecompositionOutcomeSummaryProps> = ({
  outcome,
}) => {
  const tone =
    outcome.status === 'bypassed' ? 'bypassed' : outcome.status === 'fallback' ? 'fallback' : 'neutral';
  const colors = TONE_COLORS[tone];

  const label =
    outcome.status === 'bypassed'
      ? 'Query Router: bypassed decomposition (single-intent query)'
      : outcome.status === 'fallback'
      ? 'Query Router: decomposition fallback'
      : 'Query Router: allowed decomposition';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
        Original query:{' '}
        <span style={{ color: '#e2e8f0', fontWeight: 500 }}>"{outcome.original_query}"</span>
      </div>

      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.35rem',
          fontSize: '0.72rem',
          fontWeight: 600,
          padding: '0.2rem 0.6rem',
          borderRadius: 9999,
          width: 'fit-content',
          color: colors.color,
          background: colors.background,
          border: `1px solid ${colors.border}`,
        }}
      >
        <GitBranch size={11} />
        {label}
      </div>

      {outcome.status === 'bypassed' && (
        <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
          The router determined this query was confidently single-intent and skipped the
          decomposer entirely -- the original query above was used directly for retrieval.
        </div>
      )}

      {outcome.status === 'not_needed' && (
        <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
          The decomposer ran (the router did not bypass it) and determined a single
          information need, so retrieval ran against the original query only.
        </div>
      )}

      {outcome.status === 'fallback' && (
        <div style={{ fontSize: '0.78rem', color: '#fbbf24' }}>
          {outcome.reason ||
            'Automatic query decomposition was unavailable for this request; the original query was used directly.'}
        </div>
      )}

      {outcome.status === 'decomposed' &&
        (() => {
          // `sub_queries` always includes the original query (the decomposer
          // guarantees it is present, first, and never dropped by
          // deduplication) -- it is already shown above as "Original
          // query", so it must not also be listed here as if it were an
          // additional, useful subquery the decomposer came up with.
          const additionalSubQueries = outcome.sub_queries.filter(
            (subQuery) => subQuery.trim().toLowerCase() !== outcome.original_query.trim().toLowerCase()
          );

          return (
            <>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                The decomposer split this query into the original query plus{' '}
                <strong style={{ color: '#818cf8' }}>{additionalSubQueries.length}</strong>{' '}
                additional retrieval sub-{additionalSubQueries.length === 1 ? 'query' : 'queries'},
                each run independently and merged:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {additionalSubQueries.map((subQuery, idx) => (
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
          );
        })()}
    </div>
  );
};
