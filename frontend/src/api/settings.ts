/**
 * API client for the user-facing system-instruction setting -- the only
 * module allowed to call GET/PUT /settings/system-instruction. This
 * instruction is prepended to the LLM's system prompt on every
 * POST /research/ask call, ahead of the retrieved-chunk context (see
 * app/generation/openrouter.py).
 *
 * The backend also has a separate DEFAULT instruction layer
 * (GET/PUT /settings/default-system-instruction), combined ahead of this
 * one as [default] + [this instruction] + [built-in prompt]. It is
 * deliberately NOT exposed here or anywhere in the UI -- it is an
 * operator-set baseline, not an end-user-facing setting. Set it directly
 * via that endpoint (e.g. curl) or by editing its file, not through this
 * app.
 */
import { SystemInstructionResponse } from '../types/settings';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status} (${response.statusText})`;
    try {
      const errorData = await response.json();
      if (typeof errorData?.detail === 'string') {
        errorMessage = errorData.detail;
      } else if (errorData?.detail) {
        // FastAPI's validation-error shape (e.g. the 5000-character cap) is
        // a list of objects, not a plain string -- stringify rather than
        // showing "[object Object]".
        errorMessage = JSON.stringify(errorData.detail);
      }
    } catch {
      // Fall back to status text.
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Fetches the currently configured system instruction. Backend route:
 * GET /settings/system-instruction. `text` is "" when none is configured.
 */
export async function getSystemInstruction(): Promise<SystemInstructionResponse> {
  const response = await fetch(`${API_BASE}/settings/system-instruction`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  return handleResponse<SystemInstructionResponse>(response);
}

/**
 * Persists a new system instruction, replacing whatever was there before.
 * Backend route: PUT /settings/system-instruction. Pass "" to clear it.
 */
export async function updateSystemInstruction(
  text: string
): Promise<SystemInstructionResponse> {
  const response = await fetch(`${API_BASE}/settings/system-instruction`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ text }),
  });
  return handleResponse<SystemInstructionResponse>(response);
}
