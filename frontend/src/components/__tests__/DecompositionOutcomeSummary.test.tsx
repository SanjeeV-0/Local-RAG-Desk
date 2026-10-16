import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { DecompositionOutcomeSummary } from '../DecompositionOutcomeSummary';
import { DecompositionOutcome } from '../../types/research';

afterEach(() => {
  cleanup();
});

describe('DecompositionOutcomeSummary', () => {
  it('shows the original query separately from additional subqueries, without duplicating it', () => {
    const outcome: DecompositionOutcome = {
      original_query: 'What is RAG and how does reranking work?',
      // The decomposer always includes the original query first -- the
      // component must not also list it as an "additional" subquery.
      sub_queries: [
        'What is RAG and how does reranking work?',
        'What is RAG?',
        'How does reranking work?',
      ],
      status: 'decomposed',
      reason: null,
    };

    const { container } = render(<DecompositionOutcomeSummary outcome={outcome} />);

    // The additional subqueries are shown as their own chips...
    expect(screen.getByText('What is RAG?')).toBeInTheDocument();
    expect(screen.getByText('How does reranking work?')).toBeInTheDocument();

    // ...but the full original query string is rendered exactly once overall
    // (in the "Original query" line) -- never a second time as though it
    // were itself one of the additional decomposed subqueries.
    const occurrences = container.textContent?.split(outcome.original_query).length! - 1;
    expect(occurrences).toBe(1);

    // Count reflects only the additional subqueries, not the original.
    expect(screen.getByText('2')).toBeInTheDocument();
  });

  it('is case-insensitive when excluding the original query from the subquery list', () => {
    const outcome: DecompositionOutcome = {
      original_query: 'What is RAG?',
      sub_queries: ['what is rag?', 'How does reranking work?'],
      status: 'decomposed',
      reason: null,
    };

    render(<DecompositionOutcomeSummary outcome={outcome} />);

    expect(screen.queryByText('what is rag?')).not.toBeInTheDocument();
    expect(screen.getByText('How does reranking work?')).toBeInTheDocument();
  });

  it('does not fabricate a decomposed status when the outcome says bypassed', () => {
    const outcome: DecompositionOutcome = {
      original_query: 'What is RAG?',
      sub_queries: ['What is RAG?'],
      status: 'bypassed',
      reason: null,
    };

    render(<DecompositionOutcomeSummary outcome={outcome} />);

    expect(screen.getByText(/bypassed decomposition/i)).toBeInTheDocument();
    expect(screen.queryByText(/additional retrieval sub-/i)).not.toBeInTheDocument();
  });

  it('renders the safe fallback reason as provided, never inventing one when present', () => {
    const outcome: DecompositionOutcome = {
      original_query: 'What is RAG?',
      sub_queries: ['What is RAG?'],
      status: 'fallback',
      reason: 'Automatic query decomposition was unavailable for this request; the original query was used directly.',
    };

    render(<DecompositionOutcomeSummary outcome={outcome} />);

    expect(
      screen.getByText(
        'Automatic query decomposition was unavailable for this request; the original query was used directly.',
      ),
    ).toBeInTheDocument();
  });
});
