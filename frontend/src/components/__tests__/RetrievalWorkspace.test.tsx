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
  vi.clearAllMocks();
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

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);

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

describe('RetrievalWorkspace Query Router field', () => {
  it('sends the current router_enabled boolean on the request', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={true} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).toMatchObject({
      query: 'a query',
      router_enabled: true,
    });
  });
});

describe('RetrievalWorkspace router decision display', () => {
  it('shows "allowed decomposition" and the actual subqueries when decomposition ran', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValueOnce({
      ...SUCCESSFUL_RESPONSE,
      trace_requested: true,
      trace_available: true,
      trace: {
        query: 'complex question',
        original_query: 'complex question',
        sub_queries: ['sub query alpha', 'sub query beta'],
        candidate_limit: 50,
        raw_candidate_count: 4,
        deduplicated_candidate_count: 3,
        candidates: [],
        final_results: [],
        context: null,
        router_bypassed: false,
      },
    });

    render(<RetrievalWorkspace traceEnabled={true} routerEnabled={true} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'complex question');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByText('sub query alpha')).toBeInTheDocument());
    expect(screen.getByText('sub query beta')).toBeInTheDocument();
    expect(screen.getByText(/Query Router: allowed decomposition/i)).toBeInTheDocument();
  });

  it('shows a distinct "bypassed" decision and does not claim decomposition ran', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValueOnce({
      ...SUCCESSFUL_RESPONSE,
      trace_requested: true,
      trace_available: true,
      trace: {
        query: 'simple question',
        original_query: 'simple question',
        sub_queries: ['simple question'],
        candidate_limit: 50,
        raw_candidate_count: 1,
        deduplicated_candidate_count: 1,
        candidates: [],
        final_results: [],
        context: null,
        router_bypassed: true,
      },
    });

    render(<RetrievalWorkspace traceEnabled={true} routerEnabled={true} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'simple question');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() =>
      expect(screen.getByText(/Query Router: bypassed decomposition/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText(/allowed decomposition/i)).not.toBeInTheDocument();
  });

  it('does not invent a router decision when trace is unavailable', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValueOnce(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByText('Some retrieved content')).toBeInTheDocument());
    expect(screen.queryByText(/Query Router:/i)).not.toBeInTheDocument();
  });
});

describe('RetrievalWorkspace request cancellation', () => {
  it('cancelling shows a cancellation state, not a generic error, and clears loading', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);

    let capturedSignal: AbortSignal | undefined;
    mocked.mockImplementationOnce(
      (_params, signal) =>
        new Promise((_resolve, reject) => {
          capturedSignal = signal;
          signal?.addEventListener('abort', () => {
            const err = new Error('Aborted');
            err.name = 'AbortError';
            reject(err);
          });
        }),
    );

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByRole('button', { name: /^Cancel$/i })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /^Cancel$/i }));

    expect(capturedSignal?.aborted).toBe(true);

    await waitFor(() => expect(screen.getByText('Request Cancelled')).toBeInTheDocument());
    expect(screen.queryByText('Retrieval API Error')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Execute Vector Retrieval/i })).not.toBeDisabled();
  });
});
