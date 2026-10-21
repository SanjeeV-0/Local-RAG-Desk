import { describe, it, expect, vi, afterEach } from 'vitest';
import { getSystemInstruction, updateSystemInstruction } from '../settings';

function mockFetchOnce(response: { status: number; statusText?: string; body: unknown }) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: response.status >= 200 && response.status < 300,
    status: response.status,
    statusText: response.statusText ?? '',
    json: async () => response.body,
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getSystemInstruction', () => {
  it('sends a GET request and returns the parsed response', async () => {
    const fetchMock = mockFetchOnce({ status: 200, body: { text: 'Be concise.' } });

    const result = await getSystemInstruction();

    expect(result).toEqual({ text: 'Be concise.' });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/settings/system-instruction'),
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('throws a readable error on a plain string detail', async () => {
    mockFetchOnce({ status: 500, body: { detail: 'Something went wrong.' } });

    await expect(getSystemInstruction()).rejects.toThrow('Something went wrong.');
  });
});

describe('updateSystemInstruction', () => {
  it('sends a PUT request with the text in the JSON body', async () => {
    const fetchMock = mockFetchOnce({ status: 200, body: { text: 'Answer formally.' } });

    const result = await updateSystemInstruction('Answer formally.');

    expect(result).toEqual({ text: 'Answer formally.' });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/settings/system-instruction'),
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({ text: 'Answer formally.' }),
      }),
    );
  });

  it('stringifies a non-string validation-error detail instead of showing [object Object]', async () => {
    mockFetchOnce({
      status: 422,
      body: {
        detail: [{ loc: ['body', 'text'], msg: 'String should have at most 5000 characters' }],
      },
    });

    await expect(updateSystemInstruction('x'.repeat(5001))).rejects.toThrow(
      /String should have at most 5000 characters/,
    );
  });
});
