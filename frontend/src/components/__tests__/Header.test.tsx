import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from '../Header';

afterEach(cleanup);

function renderHeader(onToggleTrace = vi.fn(), traceEnabled = false) {
  render(
    <Header
      activeTab="research"
      onSelectTab={vi.fn()}
      traceEnabled={traceEnabled}
      onToggleTrace={onToggleTrace}
      isBackendHealthy={true}
      readinessInfo={{ status: 'ready', checks: { database: true } }}
      hasTraceData={false}
    />,
  );
  return onToggleTrace;
}

describe('Header Trace Mode toggle keyboard activation', () => {
  it('is a real button, reachable by Tab, and exposes its pressed state', () => {
    renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Trace Mode' });
    expect(toggle.tagName).toBe('BUTTON');
    expect(toggle).toHaveAttribute('aria-pressed', 'false');
  });

  it('toggles on Enter when focused', async () => {
    const user = userEvent.setup();
    const onToggleTrace = renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Trace Mode' });
    toggle.focus();
    expect(toggle).toHaveFocus();

    await user.keyboard('{Enter}');

    expect(onToggleTrace).toHaveBeenCalledTimes(1);
    expect(onToggleTrace).toHaveBeenCalledWith(true);
  });

  it('toggles on Space when focused, exactly once (no double-toggle with click)', async () => {
    const user = userEvent.setup();
    const onToggleTrace = renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Trace Mode' });
    toggle.focus();

    await user.keyboard(' ');

    expect(onToggleTrace).toHaveBeenCalledTimes(1);
    expect(onToggleTrace).toHaveBeenCalledWith(true);
  });

  it('still toggles on mouse click', async () => {
    const user = userEvent.setup();
    const onToggleTrace = renderHeader();

    await user.click(screen.getByRole('button', { name: 'Toggle Trace Mode' }));

    expect(onToggleTrace).toHaveBeenCalledTimes(1);
    expect(onToggleTrace).toHaveBeenCalledWith(true);
  });

  it('reflects aria-pressed=true when traceEnabled is true', () => {
    renderHeader(vi.fn(), true);

    expect(screen.getByRole('button', { name: 'Toggle Trace Mode' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });
});
