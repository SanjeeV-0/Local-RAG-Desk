/**
 * API client for the Research & Synthesis (RAG) and Retrieval Explorer
 * pages -- the only module allowed to call POST /research/ask and
 * POST /retrieval/search. The frontend never re-implements decomposition,
 * merging, or reranking itself; these two functions just forward the
 * caller's request and return the backend's response verbatim.
 */
import {
  ResearchRequest,
  ResearchResponse,
  RetrievalSearchRequest,
  RetrievalSearchResponse,
  BackendHealthResponse,
  BackendReadinessResponse,
} from '../types/research';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || '';

/**
 * Helper to handle fetch responses and extract meaningful FastAPI error details
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `HTTP ${response.status} (${response.statusText})`;
    try {
      const errorData = await response.json();
      if (errorData?.detail) {
        errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : JSON.stringify(errorData.detail);
      }
    } catch {
      // Fall back to status text
    }
    throw new Error(errorMessage);
  }
  return response.json();
}

/**
 * Sends a research question request to backend POST /research/ask.
 *
 * `routerEnabled` is sent as the request's explicit `router_enabled` --
 * the frontend's Query Router toggle is binary (no "inherit the server
 * default" UI state), so it is always a concrete boolean here, never
 * omitted/null. `signal`, when provided, lets the caller abort the
 * in-flight fetch (e.g. a Cancel button or a superseding new request) --
 * an aborted fetch rejects with a `DOMException` named `"AbortError"`,
 * which is deliberately NOT rewritten below so callers can distinguish a
 * user-initiated cancellation from a real network/backend failure.
 */
export async function askResearchQuestion(
  query: string,
  trace: boolean = false,
  routerEnabled: boolean = false,
  signal?: AbortSignal
): Promise<ResearchResponse> {
  const payload: ResearchRequest = {
    query: query.trim(),
    trace,
    router_enabled: routerEnabled,
  };

  try {
    const response = await fetch(`${API_BASE}/research/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    return await handleResponse<ResearchResponse>(response);
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw error;
    }
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      throw new Error(
        'Unable to connect to the research agent backend. Please ensure the FastAPI server is running on http://localhost:8000.'
      );
    }
    throw error;
  }
}

/**
 * Directly queries the vector retrieval endpoint POST /retrieval/search.
 *
 * `params.trace` is sent as-is -- the caller (RetrievalWorkspace) is
 * responsible for passing the actual Trace Mode toggle state here so the
 * UI's displayed mode and the request it sends can never drift apart. The
 * backend still applies its own `TRACE_ENABLED` ceiling on top of this.
 */
export async function searchRetrievedChunks(
  params: RetrievalSearchRequest,
  signal?: AbortSignal
): Promise<RetrievalSearchResponse> {
  const payload: Record<string, unknown> = {
    query: params.query.trim(),
    limit: params.limit ?? 10,
    trace: params.trace ?? false,
    // Same explicit-boolean contract as `askResearchQuestion` -- the
    // Retrieval Explorer's Query Router toggle always sends a concrete
    // boolean, defaulting to false (not sent at all) only if the caller
    // genuinely omits it.
    router_enabled: params.router_enabled ?? false,
  };

  if (params.document_id) {
    payload.document_id = params.document_id;
  }
  if (params.section_id) {
    payload.section_id = params.section_id;
  }
  // Omitted (not even sent as `null`) when unset, so the server applies its
  // own default -- never invent a numeric value here.
  if (params.candidate_limit !== undefined && params.candidate_limit !== null) {
    payload.candidate_limit = params.candidate_limit;
  }

  try {
    const response = await fetch(`${API_BASE}/retrieval/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    return await handleResponse<RetrievalSearchResponse>(response);
  } catch (error: any) {
    if (error?.name === 'AbortError') {
      throw error;
    }
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      throw new Error(
        'Unable to connect to the retrieval service. Please verify the backend is running.'
      );
    }
    throw error;
  }
}

/**
 * Checks backend general health: GET /health
 */
export async function checkBackendHealth(): Promise<BackendHealthResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Checks backend database readiness: GET /health/ready
 */
export async function checkBackendReadiness(): Promise<BackendReadinessResponse | null> {
  try {
    const response = await fetch(`${API_BASE}/health/ready`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}
