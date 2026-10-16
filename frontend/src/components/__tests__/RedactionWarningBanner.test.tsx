import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import { RedactionWarningBanner } from '../RedactionWarningBanner';
import { RedactionOutcome } from '../../types/research';

afterEach(() => {
  cleanup();
});

const CLEAN: RedactionOutcome = {
  redaction_applied: false,
  affected_source_count: 0,
  answer_redacted: false,
  possibly_incomplete: false,
};

describe('RedactionWarningBanner', () => {
  it('renders nothing when the response is not possibly incomplete', () => {
    const { container } = render(<RedactionWarningBanner redaction={CLEAN} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('announces redacted sources with an accessible status role', () => {
    render(
      <RedactionWarningBanner
        redaction={{
          redaction_applied: true,
          affected_source_count: 2,
          answer_redacted: false,
          possibly_incomplete: true,
        }}
      />,
    );

    const status = screen.getByRole('status');
    expect(status).toHaveTextContent('Some content was withheld');
    expect(status).toHaveTextContent('2 sources contained sensitive information');
  });

  it('uses singular phrasing for exactly one affected source', () => {
    render(
      <RedactionWarningBanner
        redaction={{
          redaction_applied: true,
          affected_source_count: 1,
          answer_redacted: false,
          possibly_incomplete: true,
        }}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent('1 source contained sensitive information');
  });

  it('mentions the answer itself when answer_redacted is true', () => {
    render(
      <RedactionWarningBanner
        redaction={{
          redaction_applied: true,
          affected_source_count: 0,
          answer_redacted: true,
          possibly_incomplete: true,
        }}
      />,
    );

    expect(screen.getByRole('status')).toHaveTextContent('part of the generated answer was redacted');
  });

  it('mentions both sources and the answer when both are redacted', () => {
    render(
      <RedactionWarningBanner
        redaction={{
          redaction_applied: true,
          affected_source_count: 3,
          answer_redacted: true,
          possibly_incomplete: true,
        }}
      />,
    );

    const status = screen.getByRole('status');
    expect(status).toHaveTextContent('3 sources contained sensitive information');
    expect(status).toHaveTextContent('part of the generated answer was redacted');
  });

  it('renders nothing for a malformed contract (possibly_incomplete true but nothing redacted)', () => {
    // Regression: `possibly_incomplete` previously got set to true whenever
    // the overall response was partial for ANY reason (e.g. a generation
    // failure), not just when something was actually redacted -- producing
    // a stray, contentless ". This response may be incomplete..." sentence
    // with no source count. The backend contract is now fixed, but this
    // component must also never render that malformed sentence even if a
    // future caller regresses the same way.
    const { container } = render(
      <RedactionWarningBanner
        redaction={{
          redaction_applied: false,
          affected_source_count: 0,
          answer_redacted: false,
          possibly_incomplete: true,
        }}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it('never renders a stray leading period with no content for any redaction state', () => {
    const states: RedactionOutcome[] = [
      CLEAN,
      { redaction_applied: true, affected_source_count: 1, answer_redacted: false, possibly_incomplete: true },
      { redaction_applied: true, affected_source_count: 2, answer_redacted: false, possibly_incomplete: true },
      { redaction_applied: false, affected_source_count: 0, answer_redacted: false, possibly_incomplete: true },
    ];

    for (const redaction of states) {
      const { container, unmount } = render(<RedactionWarningBanner redaction={redaction} />);
      const text = container.textContent ?? '';
      expect(text.startsWith('.')).toBe(false);
      unmount();
    }
  });
});
