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
