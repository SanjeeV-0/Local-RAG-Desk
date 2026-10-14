import { useCallback, useEffect, useRef } from 'react';

export interface ActiveRequestHandle {
  /** Pass straight through to `fetch`/the API wrapper's `signal` param. */
  signal: AbortSignal;
  /**
   * Call this before applying a resolved/rejected result to state. Returns
   * false if a NEWER request has since been started (or the component has
   * unmounted and started a fresh instance) for this same workflow -- in
   * that case the caller must not touch state, since a more recent request
   * already owns it.
   */
  isCurrent: () => boolean;
}

/**
 * Tracks the single "current" in-flight request for one workflow (the
 * Research tab, the Retrieval Explorer) so that:
 *
 *  - starting a new request automatically aborts whatever was previously
 *    in flight for the same workflow -- only the latest request's
 *    AbortController is ever live, so an old request can never finish
 *    after, and overwrite the state of, a newer one;
 *  - a response that still manages to resolve/reject after being
 *    superseded (or after an explicit user cancel) can check `isCurrent()`
 *    before touching state, so it never overwrites what a newer request
 *    (or an intentional cancellation) already established;
 *  - unmounting the owning component aborts whatever request is still in
 *    flight, so no `setState` call ever fires after unmount.
 *
 * This hook holds no application state itself (no loading/error/result) --
 * it only owns the abort/identity bookkeeping. The caller's own
 * try/catch/finally still decides what each outcome means for its UI.
 */
export function useCancellableRequest() {
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const start = useCallback((): ActiveRequestHandle => {
    // A new request supersedes whatever was previously in flight for this
    // workflow -- abort it rather than let two requests race to set state.
    controllerRef.current?.abort();

    const controller = new AbortController();
    controllerRef.current = controller;

    const requestId = ++requestIdRef.current;

    return {
      signal: controller.signal,
      isCurrent: () => requestIdRef.current === requestId,
    };
  }, []);

  /** Aborts whatever request is currently in flight (e.g. a Cancel button). */
  const cancel = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  return { start, cancel };
}
