import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RetrievalWorkspace } from '../RetrievalWorkspace';
import { searchRetrievedChunks } from '../../api/research';
import { RetrievalSearchResponse } from '../../types/research';

vi.mock('../../api/research', () => ({
  searchRetrievedChunks: vi.fn(),
}));

afterEach(() => {
  cleanup();
});

const SUCCESSFUL_RESPONSE: RetrievalSearchResponse = {
  results: [
    {
      document_id: 'doc-1',
      chunk_id: 'chunk-1',
      section_id: 'section-1',
      section_path: 'Intro',
      page_numbers: [1],
      content: 'Some retrieved content',
      distance: 0.12,
      similarity: 0.88,
      rerank_score: 0.75,
    },
  ],
  trace_requested: false,
  trace_available: false,
  trace_unavailable_reason: null,
  trace: null,
};

describe('RetrievalWorkspace stale-result handling', () => {
  it('clears a previous successful response when a subsequent request fails', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValueOnce(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'first query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByText('Some retrieved content')).toBeInTheDocument());

    mocked.mockRejectedValueOnce(new Error('backend exploded'));

    await user.clear(textarea);
    await user.type(textarea, 'second query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByText('backend exploded')).toBeInTheDocument());

    // The stale chunk from the FIRST (successful) request must not still
    // be displayed next to the error from the second (failed) request.
    expect(screen.queryByText('Some retrieved content')).not.toBeInTheDocument();
  });
});
