import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ResearchPage } from '../ResearchPage';
import {
  askResearchQuestion,
  checkBackendHealth,
  checkBackendReadiness,
} from '../../api/research';

vi.mock('../../api/research', () => ({
  askResearchQuestion: vi.fn(),
  checkBackendHealth: vi.fn(),
  checkBackendReadiness: vi.fn(),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ResearchPage stale-result handling', () => {
  it('clears a previous successful answer when a subsequent research request fails', async () => {
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });

    const user = userEvent.setup();
    vi.mocked(askResearchQuestion).mockResolvedValueOnce({
      answer: 'The first answer content',
      model: 'test-model',
      sources: [],
      decomposition: {
        original_query: 'first question',
        sub_queries: ['first question'],
        status: 'not_needed',
        reason: null,
      },
      trace_requested: false,
      trace_available: false,
      trace_unavailable_reason: null,
      trace: null,
    });

    render(<ResearchPage />);

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'first question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() =>
      expect(screen.getByText('The first answer content')).toBeInTheDocument(),
    );

    vi.mocked(askResearchQuestion).mockRejectedValueOnce(new Error('upstream failure'));

    await user.clear(textarea);
    await user.type(textarea, 'second question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByText('upstream failure')).toBeInTheDocument());

    // The first request's answer must not still be shown as though it
    // belongs to the second (failed) request.
    expect(screen.queryByText('The first answer content')).not.toBeInTheDocument();
  });
});

const BASE_RESPONSE = {
  answer: 'An answer',
  model: 'test-model',
  sources: [],
  decomposition: {
    original_query: 'a query',
    sub_queries: ['a query'],
    status: 'not_needed' as const,
    reason: null,
  },
  trace_requested: false,
  trace_available: false,
  trace_unavailable_reason: null,
  trace: null,
};

describe('ResearchPage Query Router toggle', () => {
  it('defaults OFF and sends router_enabled=false until toggled, then true', async () => {
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });
    vi.mocked(askResearchQuestion).mockResolvedValue(BASE_RESPONSE);

    const user = userEvent.setup();
    render(<ResearchPage />);

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'first question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(askResearchQuestion).toHaveBeenCalledTimes(1));
    expect(askResearchQuestion).toHaveBeenNthCalledWith(
      1,
      'first question',
      false,
      false,
      expect.anything(),
    );

    await user.click(screen.getByRole('button', { name: 'Toggle Query Router' }));

    await user.clear(textarea);
    await user.type(textarea, 'second question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(askResearchQuestion).toHaveBeenCalledTimes(2));
    expect(askResearchQuestion).toHaveBeenNthCalledWith(
      2,
      'second question',
      false,
      true,
      expect.anything(),
    );
  });
});

describe('ResearchPage decomposition outcome (no Trace Mode)', () => {
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
      vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
      vi.mocked(checkBackendReadiness).mockResolvedValue({
        status: 'ready',
        checks: { database: true },
      });

      const user = userEvent.setup();
      vi.mocked(askResearchQuestion).mockResolvedValueOnce({
        ...BASE_RESPONSE,
        decomposition: {
          original_query: 'a question',
          sub_queries: status === 'decomposed' ? ['sub a', 'sub b'] : ['a question'],
          status,
          reason,
        },
      });

      render(<ResearchPage />);

      const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
      await user.type(textarea, 'a question');
      await user.click(screen.getByRole('button', { name: /^Research$/i }));

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

describe('ResearchPage trace router-decision display', () => {
  it('renders router_bypassed and actual subqueries from the real trace contract', async () => {
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });
    vi.mocked(askResearchQuestion).mockResolvedValueOnce({
      ...BASE_RESPONSE,
      decomposition: {
        original_query: 'complex question',
        sub_queries: ['sub query one', 'sub query two'],
        status: 'decomposed',
        reason: null,
      },
      trace_requested: true,
      trace_available: true,
      trace: {
        query: 'complex question',
        original_query: 'complex question',
        sub_queries: ['sub query one', 'sub query two'],
        candidate_limit: 50,
        raw_candidate_count: 4,
        deduplicated_candidate_count: 3,
        candidates: [],
        final_results: [],
        context: null,
        router_bypassed: false,
      },
    });

    const user = userEvent.setup();
    render(<ResearchPage />);

    // Enable Trace Mode so the inline TracePanel renders.
    await user.click(screen.getByRole('button', { name: 'Toggle Trace Mode' }));

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'complex question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByText('sub query one')).toBeInTheDocument());
    expect(screen.getByText('sub query two')).toBeInTheDocument();
    expect(screen.getByText(/Query Router: allowed decomposition/i)).toBeInTheDocument();
  });

  it('renders a bypassed decision distinctly, without ever showing invented subqueries', async () => {
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });
    vi.mocked(askResearchQuestion).mockResolvedValueOnce({
      ...BASE_RESPONSE,
      decomposition: {
        original_query: 'simple question',
        sub_queries: ['simple question'],
        status: 'bypassed',
        reason: null,
      },
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

    const user = userEvent.setup();
    render(<ResearchPage />);

    await user.click(screen.getByRole('button', { name: 'Toggle Trace Mode' }));

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'simple question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() =>
      expect(screen.getByText(/Query Router: bypassed decomposition/i)).toBeInTheDocument(),
    );
    // Must never claim decomposition ran when it was bypassed.
    expect(screen.queryByText(/allowed decomposition/i)).not.toBeInTheDocument();
  });

  it('shows the decomposition outcome even when trace is unavailable, without inventing trace diagnostics', async () => {
    // This is the exact regression this milestone corrects: the Query
    // Router's decision must be visible without Trace Mode. What must
    // never appear without tracing is the richer trace-only diagnostic
    // data (candidate/dedup counts, the full TracePanel).
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });
    vi.mocked(askResearchQuestion).mockResolvedValueOnce(BASE_RESPONSE);

    const user = userEvent.setup();
    render(<ResearchPage />);

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'a question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByText('An answer')).toBeInTheDocument());

    // The decomposition outcome IS shown...
    expect(screen.getByText(/Query Router:/i)).toBeInTheDocument();
    // ...but only the compact summary -- no trace-only diagnostics appear
    // without Trace Mode (no candidate/dedup counts, no full TracePanel).
    expect(screen.queryByText('Retrieval & Execution Trace')).not.toBeInTheDocument();
    expect(screen.queryByText('Raw Candidates')).not.toBeInTheDocument();
  });

  it('does not invent a router decision or subquery list when the response has no decomposition data', async () => {
    // Defensive regression guard: if a response somehow lacked
    // `decomposition` entirely, nothing must be invented in its place.
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });
    const { decomposition: _omit, ...responseWithoutDecomposition } = BASE_RESPONSE;
    vi.mocked(askResearchQuestion).mockResolvedValueOnce(responseWithoutDecomposition as any);

    const user = userEvent.setup();
    render(<ResearchPage />);

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'a question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByText('An answer')).toBeInTheDocument());

    expect(screen.queryByText(/Query Router:/i)).not.toBeInTheDocument();
  });
});

describe('ResearchPage request cancellation', () => {
  it('cancelling shows a cancellation state, not a generic error, and clears loading', async () => {
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });

    let capturedSignal: AbortSignal | undefined;
    vi.mocked(askResearchQuestion).mockImplementationOnce(
      (_query, _trace, _router, signal) =>
        new Promise((_resolve, reject) => {
          capturedSignal = signal;
          signal?.addEventListener('abort', () => {
            const err = new Error('Aborted');
            err.name = 'AbortError';
            reject(err);
          });
        }),
    );

    const user = userEvent.setup();
    render(<ResearchPage />);

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'a question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Cancel/i }));

    expect(capturedSignal?.aborted).toBe(true);

    await waitFor(() => expect(screen.getByText('Request Cancelled')).toBeInTheDocument());
    // No generic error banner for a user-initiated cancellation.
    expect(screen.queryByText('Research Execution Error')).not.toBeInTheDocument();
    // Loading state must be cleared -- the Research button is clickable again.
    expect(screen.getByRole('button', { name: /^Research$/i })).not.toBeDisabled();
  });

  it('cancelling and immediately resubmitting shows only the second request\'s result', async () => {
    // The submit button is disabled while a request is in flight (see
    // ResearchInput), so the only user-reachable way to have a second
    // request for the same workflow is Cancel (which settles and clears
    // the first request, re-enabling submission) immediately followed by
    // a new submit. The first request's own promise is realistically
    // modeled here: aborting it rejects it with AbortError (exactly what
    // a real `fetch()` does), proving the second request's state is never
    // clobbered by the first one settling afterwards. The underlying
    // request-identity mechanism (`useCancellableRequest`) is additionally
    // proven directly, independent of this UI timing, in
    // `src/hooks/__tests__/useCancellableRequest.test.tsx`.
    vi.mocked(checkBackendHealth).mockResolvedValue({ status: 'healthy' });
    vi.mocked(checkBackendReadiness).mockResolvedValue({
      status: 'ready',
      checks: { database: true },
    });

    vi.mocked(askResearchQuestion)
      .mockImplementationOnce(
        (_query, _trace, _router, signal) =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener('abort', () => {
              const err = new Error('Aborted');
              err.name = 'AbortError';
              reject(err);
            });
          }),
      )
      .mockResolvedValueOnce({ ...BASE_RESPONSE, answer: 'Second (newer) answer' });

    const user = userEvent.setup();
    render(<ResearchPage />);

    const textarea = screen.getByPlaceholderText(/Ask a technical research question/i);
    await user.type(textarea, 'first question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /Cancel/i }));

    await waitFor(() => expect(screen.getByText('Request Cancelled')).toBeInTheDocument());

    await user.clear(textarea);
    await user.type(textarea, 'second question');
    await user.click(screen.getByRole('button', { name: /^Research$/i }));

    await waitFor(() => expect(screen.getByText('Second (newer) answer')).toBeInTheDocument());
    expect(screen.queryByText('Request Cancelled')).not.toBeInTheDocument();
  });
});
