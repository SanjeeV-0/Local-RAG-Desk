import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import { TracePanel } from '../TracePanel';
import { TraceData } from '../../types/research';

afterEach(cleanup);

const LONG_ID = 'a'.repeat(120); // an unbroken token long enough to overflow a narrow container if unwrapped

const TRACE: TraceData = {
  query: 'q',
  original_query: 'q',
  sub_queries: ['q'],
  candidate_limit: 50,
  raw_candidate_count: 1,
  deduplicated_candidate_count: 1,
  candidates: [
    {
      document_id: LONG_ID,
      chunk_id: LONG_ID,
      section_id: null,
      section_path: 'Section',
      page_numbers: [1],
      content: 'Candidate content',
      distance: 0.2,
      rerank_score: null,
    },
  ],
  final_results: [],
  context: null,
  router_bypassed: false,
};

describe('TracePanel overflow containment (DOM-level only)', () => {
  it('applies overflow-wrap to long document/chunk id spans so they are not an unbroken overflow source', () => {
    render(<TracePanel trace={TRACE} />);

    // Expand the candidate accordion to render the ID row.
    fireEvent.click(screen.getByText('Section'));

    const docIdLabel = screen.getByText(/Doc ID:/i);
    const chunkIdLabel = screen.getByText(/Chunk ID:/i);

    // This only proves the overflow-wrap declaration is actually applied to
    // the element in the DOM -- jsdom has no layout engine, so it cannot
    // confirm the long id visually stays within its container at any given
    // viewport width. That still requires manual/browser verification (see
    // Stage 2A report).
    expect(docIdLabel).toHaveStyle({ overflowWrap: 'anywhere' });
    expect(chunkIdLabel).toHaveStyle({ overflowWrap: 'anywhere' });
    expect(screen.getAllByText(LONG_ID, { exact: false }).length).toBeGreaterThan(0);
  });
});
