import React, { useState, KeyboardEvent } from 'react';
import { Search, Sparkles, Send, X } from 'lucide-react';

interface ResearchInputProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
  initialQuery?: string;
  // Shown only while `isLoading` is true. Aborts the in-flight request --
  // see ResearchPage's use of useCancellableRequest.
  onCancel?: () => void;
}

const SAMPLE_QUERIES = [
  "What does this document say about RAG?",
  "Explain vector retrieval and reranking process",
  "What parameters are used for embeddings?",
];

export const ResearchInput: React.FC<ResearchInputProps> = ({
  onSearch,
  isLoading,
  initialQuery = '',
  onCancel,
}) => {
  const [query, setQuery] = useState(initialQuery);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSearch(query.trim());
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSampleClick = (sample: string) => {
    setQuery(sample);
    onSearch(sample);
  };

  return (
    <div className="glass-card input-section">
      <form onSubmit={handleSubmit}>
        <div className="input-header-row">
          <label className="input-label" htmlFor="research-query-input">
            <Search size={16} color="#818cf8" />
            Research Question
          </label>
          <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Press <kbd style={{ background: '#1e293b', padding: '0.1rem 0.3rem', borderRadius: 4, color: '#cbd5e1' }}>Ctrl + Enter</kbd> to execute
          </span>
        </div>

        <div className="input-wrapper">
          <textarea
            id="research-query-input"
            className="research-textarea"
            placeholder="Ask a technical research question (e.g., What does this document say about retrieval augmented generation?)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            rows={3}
          />
        </div>

        <div className="input-actions-row">
          <div className="sample-prompts">
            <span style={{ fontSize: '0.75rem', color: '#64748b', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
              <Sparkles size={12} /> Try asking:
            </span>
            {SAMPLE_QUERIES.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                className="sample-prompt-btn"
                onClick={() => handleSampleClick(sample)}
                disabled={isLoading}
              >
                {sample}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            {isLoading && onCancel && (
              <button
                type="button"
                className="btn-secondary"
                onClick={onCancel}
              >
                <X size={14} />
                Cancel
              </button>
            )}
            <button
              type="submit"
              className="btn-primary"
              disabled={!query.trim() || isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner-ring" style={{ width: 16, height: 16, borderWidth: 2 }} />
                  Researching...
                </>
              ) : (
                <>
                  <Send size={16} />
                  Research
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};
