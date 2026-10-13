import { RefObject, useEffect, useRef } from 'react';

/**
 * Shared ESC-to-close, body-scroll-lock, focus-trap, and
 * focus-restore-on-close behavior for this app's hand-rolled modal dialogs
 * (`ConfirmModal`, `UploadDocumentModal`) -- there is no overlay/dialog
 * library in this app (see frontend audit), so both modals previously
 * reimplemented a subset of this independently and inconsistently (only
 * one locked body scroll / handled ESC, neither trapped or restored
 * focus). This hook is the single place that behavior lives now.
 */

interface UseModalA11yOptions {
  isOpen: boolean;
  onClose: () => void;
  // When true, ESC must not close the modal (e.g. while an upload/confirm
  // action is in flight) -- mirrors the existing disabled-while-loading
  // behavior of the close button/backdrop click in both modals.
  closeDisabled?: boolean;
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  // No `offsetParent`-based visibility filtering here: none of this app's
  // modals render visually-hidden-but-present focusable elements, and
  // `offsetParent` requires a real layout engine -- it is always `null` in
  // jsdom (no layout), which would silently break focus trapping under
  // test even though the same code works in a real browser.
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}

export function useModalA11y<T extends HTMLElement>(
  containerRef: RefObject<T>,
  { isOpen, onClose, closeDisabled = false }: UseModalA11yOptions,
): void {
  // Kept in refs (not effect dependencies) so a mid-session change to
  // `onClose`/`closeDisabled` (e.g. isUploading flipping true/false) never
  // re-runs the setup/teardown below and re-steals focus from the user.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  const closeDisabledRef = useRef(closeDisabled);
  closeDisabledRef.current = closeDisabled;

  useEffect(() => {
    if (!isOpen) return undefined;

    const triggerElement = document.activeElement as HTMLElement | null;
    const previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const container = containerRef.current;
    const initialFocusable = container ? getFocusableElements(container) : [];
    (initialFocusable[0] ?? container)?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (!closeDisabledRef.current) {
          onCloseRef.current();
        }
        return;
      }

      if (event.key !== 'Tab' || !container) return;

      const focusable = getFocusableElements(container);
      if (focusable.length === 0) {
        // Nothing to focus inside the modal -- keep focus from escaping.
        event.preventDefault();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;

      if (event.shiftKey) {
        if (active === first || !container.contains(active)) {
          event.preventDefault();
          last.focus();
        }
      } else if (active === last || !container.contains(active)) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousBodyOverflow;
      triggerElement?.focus?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, containerRef]);
}
