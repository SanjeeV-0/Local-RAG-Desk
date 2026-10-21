import React, { useEffect, useRef, useState } from 'react';
import { MessageSquareText, X, AlertTriangle, Check } from 'lucide-react';
import { getSystemInstruction, updateSystemInstruction } from '../api/settings';
import { useModalA11y } from '../hooks/useModalA11y';

interface SystemInstructionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const MAX_LENGTH = 5000;

// There is also a DEFAULT system-instruction layer
// (GET/PUT /settings/default-system-instruction) that this modal
// deliberately does NOT expose -- it is an operator-set baseline, not an
// end-user-facing setting, and is combined with this one on the backend as
// [default] + [this instruction] + [built-in prompt] (see
// app/generation/openrouter.py). Set it directly via that endpoint or by
// editing its file, not through the UI.
export const SystemInstructionModal: React.FC<SystemInstructionModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedJustNow, setSavedJustNow] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);

  // Re-fetches once per mount. This modal is never kept mounted across a
  // close/reopen cycle (it returns `null` entirely while closed, below),
  // so every reopen is a fresh mount, which naturally re-runs this effect
  // -- no explicit reload tracking needed.
  useEffect(() => {
    if (!isOpen) return undefined;

    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setSavedJustNow(false);

    getSystemInstruction()
      .then((response) => {
        if (cancelled) return;
        setText(response.text);
      })
      .catch((err: any) => {
        if (cancelled) return;
        setError(err?.message || 'Failed to load the current system instruction.');
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  const handleClose = () => {
    if (isSaving) return;
    onClose();
  };

  useModalA11y(modalRef, {
    isOpen,
    onClose: handleClose,
    closeDisabled: isSaving,
  });

  if (!isOpen) return null;

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    setSavedJustNow(false);

    try {
      const response = await updateSystemInstruction(text);
      setText(response.text);
      setSavedJustNow(true);
    } catch (err: any) {
      setError(err?.message || 'Failed to save the system instruction.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div
        className="modal-container glass-card"
        ref={modalRef}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="system-instruction-modal-title"
      >
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div className="modal-icon-badge">
              <MessageSquareText size={20} />
            </div>
            <h3 id="system-instruction-modal-title" className="modal-title">
              System Instruction
            </h3>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={handleClose}
            disabled={isSaving}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <p className="modal-description">
            Sent to the LLM ahead of its built-in instructions on every research
            question -- and therefore always before it reads any retrieved document
            chunks, which only appear later in the prompt. Applies globally to every
            question, immediately after saving (no restart needed). Leave blank to use
            the default behavior only.
          </p>

          {isLoading ? (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '1.5rem 0',
                color: '#94a3b8',
              }}
            >
              <span className="spinner-ring" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Loading current instruction...
            </div>
          ) : (
            <>
              <textarea
                className="research-textarea"
                rows={6}
                maxLength={MAX_LENGTH}
                placeholder="e.g. Keep answers under 200 words. Always cite page numbers when available."
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  setSavedJustNow(false);
                }}
                disabled={isSaving}
                style={{ marginTop: '1rem', width: '100%' }}
              />
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  fontSize: '0.72rem',
                  color: '#64748b',
                  marginTop: '0.3rem',
                }}
              >
                {text.length} / {MAX_LENGTH}
              </div>
            </>
          )}

          {error && (
            <div className="error-banner" style={{ marginTop: '1rem', marginBottom: 0 }}>
              <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <div className="error-title">Request Failed</div>
                <div className="error-desc">{error}</div>
              </div>
            </div>
          )}

          {savedJustNow && !error && (
            <div
              role="status"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                marginTop: '1rem',
                fontSize: '0.8rem',
                color: '#10b981',
              }}
            >
              <Check size={14} />
              Saved -- applies to every question from now on.
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button type="button" className="btn-secondary" onClick={handleClose} disabled={isSaving}>
            Close
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={handleSave}
            disabled={isLoading || isSaving}
          >
            {isSaving ? (
              <>
                <span className="spinner-ring" style={{ width: 14, height: 14, borderWidth: 2 }} />
                Saving...
              </>
            ) : (
              'Save Instruction'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
