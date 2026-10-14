import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Header } from '../Header';

afterEach(cleanup);

function renderHeader({
  onToggleTrace = vi.fn(),
  traceEnabled = false,
  onToggleRouter = vi.fn(),
  routerEnabled = false,
}: {
  onToggleTrace?: ReturnType<typeof vi.fn>;
  traceEnabled?: boolean;
  onToggleRouter?: ReturnType<typeof vi.fn>;
  routerEnabled?: boolean;
} = {}) {
  render(
    <Header
      activeTab="research"
      onSelectTab={vi.fn()}
      traceEnabled={traceEnabled}
      onToggleTrace={onToggleTrace}
      routerEnabled={routerEnabled}
      onToggleRouter={onToggleRouter}
      isBackendHealthy={true}
      readinessInfo={{ status: 'ready', checks: { database: true } }}
      hasTraceData={false}
    />,
  );
  return { onToggleTrace, onToggleRouter };
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
    const { onToggleTrace } = renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Trace Mode' });
    toggle.focus();
    expect(toggle).toHaveFocus();

    await user.keyboard('{Enter}');

    expect(onToggleTrace).toHaveBeenCalledTimes(1);
    expect(onToggleTrace).toHaveBeenCalledWith(true);
  });

  it('toggles on Space when focused, exactly once (no double-toggle with click)', async () => {
    const user = userEvent.setup();
    const { onToggleTrace } = renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Trace Mode' });
    toggle.focus();

    await user.keyboard(' ');

    expect(onToggleTrace).toHaveBeenCalledTimes(1);
    expect(onToggleTrace).toHaveBeenCalledWith(true);
  });

  it('still toggles on mouse click', async () => {
    const user = userEvent.setup();
    const { onToggleTrace } = renderHeader();

    await user.click(screen.getByRole('button', { name: 'Toggle Trace Mode' }));

    expect(onToggleTrace).toHaveBeenCalledTimes(1);
    expect(onToggleTrace).toHaveBeenCalledWith(true);
  });

  it('reflects aria-pressed=true when traceEnabled is true', () => {
    renderHeader({ traceEnabled: true });

    expect(screen.getByRole('button', { name: 'Toggle Trace Mode' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });
});

describe('Header Query Router toggle', () => {
  it('defaults to OFF and is a real, accessible button', () => {
    renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Query Router' });
    expect(toggle.tagName).toBe('BUTTON');
    expect(toggle).toHaveAttribute('aria-pressed', 'false');
  });

  it('toggles on Enter when focused', async () => {
    const user = userEvent.setup();
    const { onToggleRouter } = renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Query Router' });
    toggle.focus();

    await user.keyboard('{Enter}');

    expect(onToggleRouter).toHaveBeenCalledTimes(1);
    expect(onToggleRouter).toHaveBeenCalledWith(true);
  });

  it('toggles on Space when focused, exactly once', async () => {
    const user = userEvent.setup();
    const { onToggleRouter } = renderHeader();

    const toggle = screen.getByRole('button', { name: 'Toggle Query Router' });
    toggle.focus();

    await user.keyboard(' ');

    expect(onToggleRouter).toHaveBeenCalledTimes(1);
    expect(onToggleRouter).toHaveBeenCalledWith(true);
  });

  it('still toggles on mouse click', async () => {
    const user = userEvent.setup();
    const { onToggleRouter } = renderHeader();

    await user.click(screen.getByRole('button', { name: 'Toggle Query Router' }));

    expect(onToggleRouter).toHaveBeenCalledTimes(1);
    expect(onToggleRouter).toHaveBeenCalledWith(true);
  });

  it('reflects aria-pressed=true when routerEnabled is true, independent of Trace Mode', () => {
    renderHeader({ routerEnabled: true, traceEnabled: false });

    expect(screen.getByRole('button', { name: 'Toggle Query Router' })).toHaveAttribute(
      'aria-pressed',
      'true',
    );
    expect(screen.getByRole('button', { name: 'Toggle Trace Mode' })).toHaveAttribute(
      'aria-pressed',
      'false',
    );
  });
});
