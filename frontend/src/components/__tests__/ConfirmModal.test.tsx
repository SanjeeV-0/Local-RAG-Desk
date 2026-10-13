import { useState } from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfirmModal } from '../ConfirmModal';

afterEach(() => {
  cleanup();
  document.body.style.overflow = '';
});

function StatefulHarness({ onCancel }: { onCancel: () => void }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setIsOpen(true)}>
        Open confirm
      </button>
      <ConfirmModal
        isOpen={isOpen}
        title="Delete item"
        description="Are you sure?"
        onConfirm={() => setIsOpen(false)}
        onCancel={() => {
          onCancel();
          setIsOpen(false);
        }}
      />
    </>
  );
}

describe('ConfirmModal accessibility behavior', () => {
  it('closes on Escape (existing behavior preserved)', () => {
    const onCancel = vi.fn();
    render(
      <ConfirmModal
        isOpen
        title="Delete item"
        description="Are you sure?"
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    );

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('does not close on Escape while isLoading (existing behavior preserved)', () => {
    const onCancel = vi.fn();
    render(
      <ConfirmModal
        isOpen
        isLoading
        title="Delete item"
        description="Are you sure?"
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    );

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onCancel).not.toHaveBeenCalled();
  });

  it('locks body scroll while open and restores it on close', () => {
    const { rerender } = render(
      <ConfirmModal
        isOpen
        title="Delete item"
        description="Are you sure?"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(document.body.style.overflow).toBe('hidden');

    rerender(
      <ConfirmModal
        isOpen={false}
        title="Delete item"
        description="Are you sure?"
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(document.body.style.overflow).toBe('');
  });

  it('restores focus to the triggering element on close', async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();

    render(<StatefulHarness onCancel={onCancel} />);

    const trigger = screen.getByRole('button', { name: 'Open confirm' });
    trigger.focus();

    await user.click(trigger);

    expect(screen.getByRole('dialog')).toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveFocus();
  });
});
