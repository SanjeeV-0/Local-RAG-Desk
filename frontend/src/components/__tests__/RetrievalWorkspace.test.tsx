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
      matched_sub_queries: ['a query'],
    },
  ],
  decomposition: {
    original_query: 'a query',
    sub_queries: ['a query'],
    status: 'not_needed',
    reason: null,
  },
  redaction: {
    redaction_applied: false,
    affected_source_count: 0,
    answer_redacted: false,
    possibly_incomplete: false,
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
        // Appears twice when decomposition produced more than one
        // sub-query: once in this always-visible summary, and again as an
        // interactive chip in the "Filter by Sub-query" control -- both are
        // legitimate, so this tolerates either rather than assuming exactly
        // one occurrence.
        expect(screen.getAllByText('sub a').length).toBeGreaterThan(0);
        expect(screen.getAllByText('sub b').length).toBeGreaterThan(0);
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

  it('sends the selected document_id as the CURRENT VERSION id, not the logical document id, and clearing the selection omits it', async () => {
    // Regression: the backend's `document_id` filter matches
    // `DocumentChunk.document_id`, a foreign key to the per-version
    // `Document.id` -- never `logical_document_id`. Sending the logical id
    // (as this component previously did) meant every filtered search
    // silently returned zero results.
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
    await user.selectOptions(select, 'version-aaa');

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).toMatchObject({ document_id: 'version-aaa' });
    // Never the logical grouping id -- that would silently match nothing.
    expect(mocked.mock.calls[0][0]).not.toMatchObject({ document_id: 'doc-aaa' });

    // Clear the selection back to "All documents" and search again.
    await user.selectOptions(select, '— All documents (no filter) —');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(2));
    expect(mocked.mock.calls[1][0]).not.toHaveProperty('document_id');
  });

  it('disables selecting a document with no current version (nothing to filter to)', async () => {
    const DOC_NO_CURRENT_VERSION: LogicalDocumentSummary = {
      logical_document_id: 'doc-no-version',
      current_version: null,
    };

    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY, DOC_NO_CURRENT_VERSION]);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await openAdvancedFilters(user);

    await waitFor(() =>
      expect(screen.getByText(/no current version -- unavailable for filtering/)).toBeInTheDocument(),
    );

    const option = screen.getByText(
      /Untitled document \(no current version -- unavailable for filtering\)/,
    ) as HTMLOptionElement;
    expect(option.disabled).toBe(true);
    expect(option.value).toBe('');
  });
});

describe('RetrievalWorkspace initialDocumentId resolution ("Search This Doc" / "Inspect Doc")', () => {
  it('resolves a logical document id ("Search This Doc") to the current version id and sends it as the filter', async () => {
    // Regression: DocumentWorkspace's "Search This Doc" button navigates
    // here with `selectedDocument.logical_document_id` (a logical id), not
    // a version id -- previously this was used as `selectedDocumentId`
    // unresolved, so the request sent a logical id as `document_id`, which
    // never matches any chunk and silently returned zero results.
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY]);
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(
      <RetrievalWorkspace
        traceEnabled={false}
        routerEnabled={false}
        initialDocumentId="doc-aaa" // DOC_READY.logical_document_id
      />,
    );

    // The dropdown must reflect the resolved document once documents load.
    await waitFor(() =>
      expect(screen.getByDisplayValue('Vector Search Fundamentals')).toBeInTheDocument(),
    );

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    // The request must carry the CURRENT VERSION id, never the logical id.
    expect(mocked.mock.calls[0][0]).toMatchObject({ document_id: 'version-aaa' });
    expect(mocked.mock.calls[0][0]).not.toMatchObject({ document_id: 'doc-aaa' });
  });

  it('still resolves a per-version id ("Inspect Doc") the same way it always did', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY]);
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(
      <RetrievalWorkspace
        traceEnabled={false}
        routerEnabled={false}
        initialDocumentId="version-aaa" // DOC_READY.current_version.id
      />,
    );

    await waitFor(() =>
      expect(screen.getByDisplayValue('Vector Search Fundamentals')).toBeInTheDocument(),
    );

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).toMatchObject({ document_id: 'version-aaa' });
  });

  it('leaves the filter cleared (never assigns a logical id as a version id) when the target has no current version', async () => {
    const DOC_NO_CURRENT_VERSION: LogicalDocumentSummary = {
      logical_document_id: 'doc-no-version',
      current_version: null,
    };

    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_NO_CURRENT_VERSION]);
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(
      <RetrievalWorkspace
        traceEnabled={false}
        routerEnabled={false}
        initialDocumentId="doc-no-version"
      />,
    );

    await waitFor(() => expect(listDocuments).toHaveBeenCalled());

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    // No safe id to filter by -- the request must not carry `document_id`
    // at all (never the logical id substituted in its place).
    expect(mocked.mock.calls[0][0]).not.toHaveProperty('document_id');
  });

  it('leaves the filter cleared when the navigation target matches no known document', async () => {
    const user = userEvent.setup();
    vi.mocked(listDocuments).mockResolvedValueOnce([DOC_READY]);
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(
      <RetrievalWorkspace
        traceEnabled={false}
        routerEnabled={false}
        initialDocumentId="some-unrelated-id"
      />,
    );

    await waitFor(() => expect(listDocuments).toHaveBeenCalled());

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked.mock.calls[0][0]).not.toHaveProperty('document_id');
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

describe('RetrievalWorkspace sub-query filter', () => {
  const DECOMPOSED_RESPONSE: RetrievalSearchResponse = {
    results: [
      {
        document_id: 'doc-1',
        chunk_id: 'chunk-a',
        section_id: 'section-1',
        section_path: 'Intro',
        page_numbers: [1],
        content: 'Content retrieved only by sub a',
        distance: 0.1,
        similarity: 0.9,
        rerank_score: 0.8,
        matched_sub_queries: ['sub a'],
      },
      {
        document_id: 'doc-1',
        chunk_id: 'chunk-b',
        section_id: 'section-2',
        section_path: 'Background',
        page_numbers: [2],
        content: 'Content retrieved only by sub b',
        distance: 0.2,
        similarity: 0.8,
        rerank_score: 0.6,
        matched_sub_queries: ['sub b'],
      },
      {
        document_id: 'doc-1',
        chunk_id: 'chunk-both',
        section_id: 'section-3',
        section_path: 'Overlap',
        page_numbers: [3],
        content: 'Content retrieved by both sub-queries',
        distance: 0.15,
        similarity: 0.85,
        rerank_score: 0.7,
        matched_sub_queries: ['sub a', 'sub b'],
      },
    ],
    decomposition: {
      original_query: 'original query',
      sub_queries: ['sub a', 'sub b'],
      status: 'decomposed',
      reason: null,
    },
    redaction: {
      redaction_applied: false,
      affected_source_count: 0,
      answer_redacted: false,
      possibly_incomplete: false,
    },
    trace_requested: false,
    trace_available: false,
    trace_unavailable_reason: null,
    trace: null,
  };

  const runSearch = async (user: ReturnType<typeof userEvent.setup>) => {
    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'original query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));
    await waitFor(() => expect(screen.getByText('Filter by Sub-query')).toBeInTheDocument());
  };

  it('does not show the filter when decomposition produced only one sub-query', async () => {
    const user = userEvent.setup();
    const mocked = vi.mocked(searchRetrievedChunks);
    mocked.mockResolvedValue(SUCCESSFUL_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);

    const textarea = screen.getByPlaceholderText(/Enter search query/i);
    await user.type(textarea, 'a query');
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));

    await waitFor(() => expect(screen.getByText('Some retrieved content')).toBeInTheDocument());
    expect(screen.queryByText('Filter by Sub-query')).not.toBeInTheDocument();
  });

  it('shows every chunk, unfiltered, before any sub-query is selected', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue(DECOMPOSED_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    expect(screen.getByText('Content retrieved only by sub a')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved only by sub b')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved by both sub-queries')).toBeInTheDocument();
  });

  it('selecting one sub-query shows only chunks it retrieved, including shared ones', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue(DECOMPOSED_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    await user.click(screen.getByRole('button', { name: 'sub a' }));

    expect(screen.getByText('Content retrieved only by sub a')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved by both sub-queries')).toBeInTheDocument();
    expect(screen.queryByText('Content retrieved only by sub b')).not.toBeInTheDocument();
  });

  it('selecting a second sub-query is additive (OR), not a replacement', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue(DECOMPOSED_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    await user.click(screen.getByRole('button', { name: 'sub a' }));
    await user.click(screen.getByRole('button', { name: 'sub b' }));

    expect(screen.getByText('Content retrieved only by sub a')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved only by sub b')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved by both sub-queries')).toBeInTheDocument();
  });

  it('deselecting a sub-query restores chunks it had filtered out', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue(DECOMPOSED_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    const subAButton = screen.getByRole('button', { name: 'sub a' });
    await user.click(subAButton);
    expect(screen.queryByText('Content retrieved only by sub b')).not.toBeInTheDocument();

    await user.click(subAButton);
    expect(screen.getByText('Content retrieved only by sub b')).toBeInTheDocument();
  });

  it('the Clear button resets the filter back to showing every chunk', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue(DECOMPOSED_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    await user.click(screen.getByRole('button', { name: 'sub a' }));
    await user.click(screen.getByRole('button', { name: /Clear/i }));

    expect(screen.getByText('Content retrieved only by sub a')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved only by sub b')).toBeInTheDocument();
    expect(screen.getByText('Content retrieved by both sub-queries')).toBeInTheDocument();
  });

  it('resets the filter selection when a new search is run', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue(DECOMPOSED_RESPONSE);

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    await user.click(screen.getByRole('button', { name: 'sub a' }));
    expect(screen.queryByText('Content retrieved only by sub b')).not.toBeInTheDocument();

    // Re-running the search (even with the same mocked response) must not
    // carry the previous selection forward -- a fresh response's results
    // should start fully unfiltered.
    await user.click(screen.getByRole('button', { name: /Execute Vector Retrieval/i }));
    await waitFor(() => expect(screen.getByText('Filter by Sub-query')).toBeInTheDocument());

    expect(screen.getByText('Content retrieved only by sub b')).toBeInTheDocument();
  });

  it('shows a distinct empty state when the filter matches no chunks', async () => {
    const user = userEvent.setup();
    vi.mocked(searchRetrievedChunks).mockResolvedValue({
      ...DECOMPOSED_RESPONSE,
      decomposition: {
        ...DECOMPOSED_RESPONSE.decomposition,
        sub_queries: ['sub a', 'sub b', 'sub c'],
      },
    });

    render(<RetrievalWorkspace traceEnabled={false} routerEnabled={false} />);
    await runSearch(user);

    // "sub c" retrieved nothing in this fixture -- selecting only it should
    // filter every chunk out, distinctly from "no chunks were retrieved at
    // all".
    await user.click(screen.getByRole('button', { name: 'sub c' }));

    expect(screen.getByText('No Chunks Match Selected Sub-queries')).toBeInTheDocument();
    expect(screen.queryByText('No Chunks Matched')).not.toBeInTheDocument();
  });
});
