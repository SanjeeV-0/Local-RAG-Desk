import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { useCancellableRequest, ActiveRequestHandle } from '../useCancellableRequest';

afterEach(cleanup);

/** Exposes the hook's return value via a ref-like mutable object so tests
 * can call `start()`/`cancel()` directly, as a real consumer component
 * would inside an event handler. */
function renderHook() {
  let api: ReturnType<typeof useCancellableRequest> | null = null;

  function Harness() {
    api = useCancellableRequest();
    return null;
  }

  const { unmount } = render(<Harness />);

  return { get: () => api!, unmount };
}

describe('useCancellableRequest', () => {
  it('starting a new request aborts the previous one and supersedes its identity', () => {
    const hook = renderHook();

    const first: ActiveRequestHandle = hook.get().start();
    expect(first.signal.aborted).toBe(false);
    expect(first.isCurrent()).toBe(true);

    const second: ActiveRequestHandle = hook.get().start();

    // Starting the second request aborts the first's signal...
    expect(first.signal.aborted).toBe(true);
    // ...and the first handle can no longer consider itself current, while
    // the second (latest) one can.
    expect(first.isCurrent()).toBe(false);
    expect(second.isCurrent()).toBe(true);
    expect(second.signal.aborted).toBe(false);
  });

  it('cancel() aborts the current request without starting a new one', () => {
    const hook = renderHook();

    const handle = hook.get().start();
    expect(handle.signal.aborted).toBe(false);

    hook.get().cancel();

    expect(handle.signal.aborted).toBe(true);
    // No new request was started, so the cancelled one is still "current"
    // -- the caller's own AbortError handling is what distinguishes an
    // intentional cancel of the current request from being superseded.
    expect(handle.isCurrent()).toBe(true);
  });

  it('aborts whatever is in flight on unmount', () => {
    const hook = renderHook();
    const handle = hook.get().start();

    expect(handle.signal.aborted).toBe(false);
    hook.unmount();
    expect(handle.signal.aborted).toBe(true);
  });
});
