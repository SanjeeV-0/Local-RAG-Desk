import { useState } from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { UploadDocumentModal } from '../UploadDocumentModal';
import { uploadDocument } from '../../api/documents';
import { DocumentUploadResult } from '../../types/document';

// uploadDocument is mocked so these tests never perform a real network
// call; its resolved/pending value is controlled per-test below.
vi.mock('../../api/documents', () => ({
  uploadDocument: vi.fn(),
  DocumentApiUnavailableError: class DocumentApiUnavailableError extends Error {},
}));

afterEach(() => {
  cleanup();
  document.body.style.overflow = '';
});

function renderModal(onClose = vi.fn()) {
  render(
    <>
      <button type="button">Trigger</button>
      <UploadDocumentModal
        isOpen
        onClose={onClose}
        onUploaded={vi.fn()}
        onViewExisting={vi.fn()}
        existingDocuments={[]}
      />
    </>,
  );
  return onClose;
}

// Wraps the modal with local open/close state so a real close (ESC,
// backdrop, button) actually unmounts it, matching how ResearchPage/
// DocumentWorkspace use it in the app.
function StatefulHarness({ onClose }: { onClose: () => void }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setIsOpen(true)}>
        Open upload
      </button>
      <UploadDocumentModal
        isOpen={isOpen}
        onClose={() => {
          onClose();
          setIsOpen(false);
        }}
        onUploaded={vi.fn()}
        onViewExisting={vi.fn()}
        existingDocuments={[]}
      />
    </>
  );
}

describe('UploadDocumentModal accessibility behavior', () => {
  it('closes on Escape', () => {
    const onClose = renderModal();

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('locks body scroll while open and restores it on close', () => {
    expect(document.body.style.overflow).toBe('');

    const { rerender } = render(
      <UploadDocumentModal
        isOpen
        onClose={vi.fn()}
        onUploaded={vi.fn()}
        onViewExisting={vi.fn()}
        existingDocuments={[]}
      />,
    );

    expect(document.body.style.overflow).toBe('hidden');

    rerender(
      <UploadDocumentModal
        isOpen={false}
        onClose={vi.fn()}
        onUploaded={vi.fn()}
        onViewExisting={vi.fn()}
        existingDocuments={[]}
      />,
    );

    expect(document.body.style.overflow).toBe('');
  });

  it('moves focus inside the modal on open and restores it to the trigger on close', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<StatefulHarness onClose={onClose} />);

    const trigger = screen.getByRole('button', { name: 'Open upload' });
    trigger.focus();
    expect(trigger).toHaveFocus();

    await user.click(trigger);

    const closeButton = screen.getByRole('button', { name: 'Close dialog' });
    expect(closeButton).toHaveFocus();

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveFocus();
  });

  it('traps Tab focus inside the modal', () => {
    renderModal();

    const dialog = screen.getByRole('dialog');
    const focusable = dialog.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled])',
    );
    const last = focusable[focusable.length - 1];

    last.focus();
    expect(last).toHaveFocus();

    fireEvent.keyDown(document, { key: 'Tab' });

    const first = focusable[0];
    expect(first).toHaveFocus();
  });

  it('does not close on Escape while an upload is in flight, and closes normally once it settles', async () => {
    let resolveUpload: (value: DocumentUploadResult) => void = () => {};
    vi.mocked(uploadDocument).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpload = resolve;
        }),
    );

    const onClose = vi.fn();
    render(<StatefulHarness onClose={onClose} />);

    await userEvent.setup().click(screen.getByRole('button', { name: 'Open upload' }));

    const file = new File(['hello world'], 'hello.txt', { type: 'text/plain' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.submit(screen.getByRole('button', { name: /Upload & Ingest/i }).closest('form')!);

    // Upload is now in flight (isUploading=true) -- ESC must not close.
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();

    resolveUpload({
      outcome: 'created',
      version: {
        id: 'v1',
        logical_document_id: 'doc1',
        version_number: 1,
        is_current: true,
        status: 'UPLOADED',
        title: 'hello',
        document_type: 'general',
        processing_attempt: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
      updated_metadata_fields: [],
    });

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });
});
