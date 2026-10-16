/**
 * Accessible warning banner shown whenever a `RedactionOutcome` indicates
 * sensitive content was withheld -- shared by the Research tab and the
 * Retrieval Explorer. Rendered ONLY when `possibly_incomplete` is true, so
 * a clean response never shows this banner at all.
 *
 * `role="status"` (not `role="alert"`) is used deliberately: this reports a
 * completed, non-urgent fact about the response that just arrived (not an
 * error requiring immediate interruption), matching how screen readers
 * should announce it -- politely, not assertively.
 */
import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { RedactionOutcome } from '../types/research';

interface RedactionWarningBannerProps {
  redaction: RedactionOutcome;
}

export const RedactionWarningBanner: React.FC<RedactionWarningBannerProps> = ({ redaction }) => {
  const parts: string[] = [];

  if (redaction.affected_source_count > 0) {
    parts.push(
      `${redaction.affected_source_count} source${
        redaction.affected_source_count === 1 ? '' : 's'
      } contained sensitive information that was redacted`
    );
  }

  if (redaction.answer_redacted) {
    parts.push('part of the generated answer was redacted');
  }

  // `possibly_incomplete` is meant to describe redaction specifically -- if
  // a caller ever sets it without a corresponding reason (no affected
  // sources, no redacted answer), there is nothing accurate to report here.
  // Render nothing rather than a sentence with no content and a stray
  // leading period (e.g. ". This response may be incomplete...").
  if (!redaction.possibly_incomplete || parts.length === 0) {
    return null;
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="warning-banner"
      style={{
        borderColor: 'rgba(245, 158, 11, 0.35)',
        background: 'rgba(245, 158, 11, 0.08)',
      }}
    >
      <ShieldAlert
        size={22}
        aria-hidden="true"
        style={{ flexShrink: 0, marginTop: 2, color: '#fbbf24' }}
      />
      <div>
        <div className="warning-title">Some content was withheld</div>
        <div className="warning-desc">
          {parts.join(' and ')}. This response may be incomplete -- information detected as
          personally identifiable (such as emails, phone numbers, SSNs, or card numbers) is
          automatically redacted and is never guaranteed to be fully recoverable from this
          answer.
        </div>
      </div>
    </div>
  );
};
