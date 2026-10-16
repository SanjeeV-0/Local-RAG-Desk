import { describe, it, expect, vi, afterEach } from 'vitest';
import { askResearchQuestion } from '../research';

function mockFetchOnce(response: { status: number; statusText?: string; body: unknown }) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      statusText: response.statusText ?? '',
      json: async () => response.body,
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('askResearchQuestion error handling', () => {
  it('renders a classified provider error as a readable message with its safe category, not raw JSON', async () => {
    mockFetchOnce({
      status: 502,
      body: {
        detail: {
          error: {
            category: 'guardrail_or_policy_rejection',
            retryable: false,
            message: "The research assistant's provider declined to process this request.",
          },
        },
      },
    });

    await expect(askResearchQuestion('What is RAG?')).rejects.toThrow(
      "The research assistant's provider declined to process this request. (guardrail_or_policy_rejection)",
    );
  });

  it('never surfaces the error as stringified JSON when it is a classified provider error', async () => {
    mockFetchOnce({
      status: 502,
      body: {
        detail: {
          error: {
            category: 'rate_limited',
            retryable: true,
            message: 'The research assistant is temporarily busy. Please try again shortly.',
          },
        },
      },
    });

    try {
      await askResearchQuestion('What is RAG?');
      throw new Error('expected askResearchQuestion to reject');
    } catch (err: any) {
      expect(err.message).not.toContain('{');
      expect(err.message).not.toContain('"category"');
    }
  });

  it('still surfaces a plain string detail as-is (unclassified FastAPI validation errors)', async () => {
    mockFetchOnce({
      status: 422,
      body: { detail: 'query must not be empty' },
    });

    await expect(askResearchQuestion('x')).rejects.toThrow('query must not be empty');
  });

  it('falls back to HTTP status text when no usable detail is present', async () => {
    mockFetchOnce({ status: 500, statusText: 'Internal Server Error', body: {} });

    await expect(askResearchQuestion('What is RAG?')).rejects.toThrow(
      'HTTP 500 (Internal Server Error)',
    );
  });
});
