import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SystemInstructionModal } from '../SystemInstructionModal';
import { getSystemInstruction, updateSystemInstruction } from '../../api/settings';

vi.mock('../../api/settings', () => ({
  getSystemInstruction: vi.fn(),
  updateSystemInstruction: vi.fn(),
}));

afterEach(() => {
  cleanup();
  document.body.style.overflow = '';
  vi.clearAllMocks();
});

function renderModal(onClose = vi.fn()) {
  render(
    <>
      <button type="button">Trigger</button>
      <SystemInstructionModal isOpen onClose={onClose} />
    </>,
  );
  return onClose;
}

describe('SystemInstructionModal', () => {
  it('loads and displays the currently configured instruction on open', async () => {
    vi.mocked(getSystemInstruction).mockResolvedValue({ text: 'Always cite page numbers.' });

    renderModal();

    await waitFor(() =>
      expect(screen.getByDisplayValue('Always cite page numbers.')).toBeInTheDocument(),
    );
  });

  it('shows an empty textarea when no instruction is configured', async () => {
    vi.mocked(getSystemInstruction).mockResolvedValue({ text: '' });

    renderModal();

    await waitFor(() =>
      expect(screen.getByPlaceholderText(/Keep answers under 200 words/i)).toBeInTheDocument(),
    );
    expect(screen.getByPlaceholderText(/Keep answers under 200 words/i)).toHaveValue('');
  });

  it('saves the edited text and shows a confirmation', async () => {
    const user = userEvent.setup();
    vi.mocked(getSystemInstruction).mockResolvedValue({ text: '' });
    vi.mocked(updateSystemInstruction).mockResolvedValue({ text: 'Be concise.' });

    renderModal();

    const textarea = await screen.findByPlaceholderText(/Keep answers under 200 words/i);
    await user.type(textarea, 'Be concise.');
    await user.click(screen.getByRole('button', { name: /Save Instruction/i }));

    await waitFor(() => expect(updateSystemInstruction).toHaveBeenCalledWith('Be concise.'));
    expect(screen.getByText(/Saved -- applies to every question/i)).toBeInTheDocument();
  });

  it('shows an error banner when loading fails, without crashing', async () => {
    vi.mocked(getSystemInstruction).mockRejectedValue(new Error('backend unreachable'));

    renderModal();

    await waitFor(() => expect(screen.getByText('backend unreachable')).toBeInTheDocument());
  });

  it('shows an error banner when saving fails, and does not show a false success message', async () => {
    const user = userEvent.setup();
    vi.mocked(getSystemInstruction).mockResolvedValue({ text: '' });
    vi.mocked(updateSystemInstruction).mockRejectedValue(new Error('save rejected'));

    renderModal();

    const textarea = await screen.findByPlaceholderText(/Keep answers under 200 words/i);
    await user.type(textarea, 'New instruction.');
    await user.click(screen.getByRole('button', { name: /Save Instruction/i }));

    await waitFor(() => expect(screen.getByText('save rejected')).toBeInTheDocument());
    expect(screen.queryByText(/Saved -- applies to every question/i)).not.toBeInTheDocument();
  });

  it('closes on Escape', async () => {
    const user = userEvent.setup();
    vi.mocked(getSystemInstruction).mockResolvedValue({ text: '' });
    const onClose = renderModal();

    await screen.findByPlaceholderText(/Keep answers under 200 words/i);
    await user.keyboard('{Escape}');

    expect(onClose).toHaveBeenCalled();
  });

  it('re-fetches the instruction every time the modal opens, never showing a stale draft', async () => {
    vi.mocked(getSystemInstruction)
      .mockResolvedValueOnce({ text: 'First.' })
      .mockResolvedValueOnce({ text: 'Second.' });

    const { rerender } = render(<SystemInstructionModal isOpen={false} onClose={vi.fn()} />);

    rerender(<SystemInstructionModal isOpen onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByDisplayValue('First.')).toBeInTheDocument());

    rerender(<SystemInstructionModal isOpen={false} onClose={vi.fn()} />);
    rerender(<SystemInstructionModal isOpen onClose={vi.fn()} />);
    await waitFor(() => expect(screen.getByDisplayValue('Second.')).toBeInTheDocument());

    expect(getSystemInstruction).toHaveBeenCalledTimes(2);
  });
});
