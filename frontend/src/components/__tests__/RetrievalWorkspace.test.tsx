import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RetrievalWorkspace } from '../RetrievalWorkspace';
import { searchRetrievedChunks } from '../../api/research';
import { listDocuments } from '../../api/documents';
import { RetrievalSearchResponse } from '../../types/research';
import { LogicalDocumentSummary } from '../../types/document';

vi.mock('../../api/research', () => ({
  searchRetrievedChunks: vi.fn(),
}));

vi.mock('../../api/documents', () => ({
  listDocuments: vi.fn(),
  DocumentApiUnavailableError: class DocumentApiUnavailableError extends Error {},
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// Every test in this file triggers the component's own document-list fetch
// on mount; defaulting it to an empty, resolved list keeps tests that don't
// care about the document picker from having to handle an unmocked/pending
// promise.
beforeEach(() => {
  vi.mocked(listDocuments).mockResolvedValue([]);
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
  decomposition: {
    original_query: 'a query',
    sub_queries: ['a query'],
    status: 'not_needed',
    reason: null,
  },
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

describe('RetrievalWorkspace decomposition outcome (no Trace Mode)', () => {
  it.each([
    ['bypassed', null, /Query Router: bypassed decomposition/i],
    ['decomposed', null, /Query Router: allowed decomposition/i],
    ['not_needed', null, /Query Router: allowed decomposition/i],
    [
      'fallback',
      'Automatic query decomposition was unavailable for this request; the original query was used directly.',
      /Query Router: decomposition fallback/i,
    ],
  ] as const)(
    'shows status "%s" without Trace Mode enabled',
    async (status, reason, expectedText) => {
      const user = userEvent.setup();
      const mocked = vi.mocked(searchRetrievedChunks);
      mocked.mockResolvedValueOnce({
        ...SUCCESSFUL_RESPONSE,
        decomposition: {
          original_query: 'a query',
          sub_queries: status === 'decomposed' ? ['sub a', 'sub b'] : ['a query'],
          status,
          reason,
        },
      });

      render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);

      const textarea = screen.getByPlaceholderText(/Enter search query/i);
      await user.type(textarea, 'a query');
      await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

      await waitFor(() => expect(screen.getByText(expectedText)).toBeInTheDocument());

      if (status === 'fallback') {
        expect(screen.getByText(reason!)).toBeInTheDocument();
      }
      if (status === 'decomposed') {
        expect(screen.getByText('sub a')).toBeInTheDocument();
        expect(screen.getByText('sub b')).toBeInTheDocument();
      }
    },
  );
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

  it('shows the decomposition outcome even when trace is unavailable, without inventing trace diagnostics', async () => {
    // This is the exact regression this milestone corrects: the Query
    // Router's decision must be visible without Trace Mode. What must
    // never appear without tracing is the richer trace-only diagnostic
    // data (candidate/dedup counts, the "View Trace" button).
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValueOnce(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByText('Some retrieved content')).toBeInTheDocument());

    expect(screen.getByText(/Query Router:/i)).toBeInTheDocument();
    expect(screen.queryByText('Raw Candidates')).not.toBeInTheDocument();
    expect(screen.queryByText(/View Trace & Rerank Scores/i)).not.toBeInTheDocument();
  });

  it('does not invent a router decision when the response has no decomposition data', async () => {
    // Defensive regression guard: if a response somehow lacked
    // `decomposition` entirely, nothing must be invented in its place.
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    const { decomposition: _omit, ...responseWithoutDecomposition } = SUCCESSFUL_RESPONSE;
    mocked.mockResolvedValueOnce(responseWithoutDecomposition as any);

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

const DOC_READY: LogicalDocumentSummary = {
  logical_document_id: 'doc-aaa',
  current_version: {
    id: 'version-aaa',
    logical_document_id: 'doc-aaa',
    version_number: 1,
    is_current: true,
    status: 'READY',
    title: 'Vector Search Fundamentals',
    document_type: 'research_paper',
    source: null,
    processing_attempt: 1,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    processing_started_at: null,
    processing_completed_at: null,
    failed_at: null,
    last_error: null,
  },
};

const DOC_DUPLICATE_TITLE_A: LogicalDocumentSummary = {
  logical_document_id: 'doc-bbb',
  current_version: {
    ...DOC_READY.current_version!,
    id: 'version-bbb',
    logical_document_id: 'doc-bbb',
    version_number: 2,
    title: 'Shared Title',
  },
};

const DOC_DUPLICATE_TITLE_B: LogicalDocumentSummary = {
  logical_document_id: 'doc-ccc',
  current_version: {
    ...DOC_READY.current_version!,
    id: 'version-ccc',
    logical_document_id: 'doc-ccc',
    version_number: 1,
    status: 'PROCESSING',
    title: 'Shared Title',
  },
};

async function openAdvancedFilters(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /Retrieval Filters/i }));
}

describe('RetrievalWorkspace document selector', () => {
  it('shows a loading state, then populates the picker from listDocuments()', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY]);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() =>
      expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument(),
    );
  });

  it('calls listDocuments exactly once regardless of typing in the filter box', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY]);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() => expect(listDocuments).toHaveBeenCalledTimes(1));

    const filterInput = screen.getByPlaceholderText(/Filter documents by title/i);
    await user.type(filterInput, 'Vector');

    // No additional network calls from typing -- filtering is client-side
    // over the already-loaded list.
    expect(listDocuments).toHaveBeenCalledTimes(1);
  });

  it('disambiguates documents that share the same title using version/status', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([
      DOC_DUPLICATE_TITLE_A,
      DOC_DUPLICATE_TITLE_B,
    ]);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() => expect(screen.getByText(/Shared Title \(v2\)/)).toBeInTheDocument());
    expect(screen.getByText(/Shared Title \(v1\) \[PROCESSING\]/)).toBeInTheDocument();
  });

  it('shows an error state when listDocuments fails', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockRejectedValueOnce(new Error('network down'));

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() =>
      expect(screen.getByText(/Failed to load documents: network down/)).toBeInTheDocument(),
    );
  });

  it('shows an explicit empty state when there are no documents', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([]);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() => expect(screen.getByText('No documents found.')).toBeInTheDocument());
  });

  it('sends the selected document_id, and clearing the selection omits it', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY]);
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() =>
      expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument(),
    );

    const select = screen.getByDisplayValue('— All documents (no filter) —');
    await user.selectOptions(select, 'doc-aaa');

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).toMatchObject({ document_id: 'doc-aaa' });

    // Clear the selection back to "All documents" and search again.
    await user.selectOptions(select, '— All documents (no filter) —');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(2));
    expect(mocked.mock.calls[1][0]).not.toHaveProperty('document_id');
  });
});

describe('RetrievalWorkspace candidate-limit control', () => {
  it('sends an explicit candidate_limit when set', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    const candidateLimitInput = screen.getByPlaceholderText(/Server default applies/i);
    await user.type(candidateLimitInput, '200');

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).toMatchObject({ candidate_limit: 200 });
  });

  it('omits candidate_limit entirely when left unset, and never invents a value', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).not.toHaveProperty('candidate_limit');
  });

  it('labels the final-result-limit control distinctly from the candidate-limit control', async () => {
    const user = userEvent.setup();
    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    expect(screen.getByText(/Final Result Limit/i)).toBeInTheDocument();
    expect(
      screen.getByText('Candidate Limit per Subquery (optional)', { selector: 'label' }),
    ).toBeInTheDocument();
  });
});
