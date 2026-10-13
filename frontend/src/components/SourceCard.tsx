import React, { useState } from 'react';
import { Source } from '../types/research';
import { BookOpen, ChevronDown, ChevronUp, FileText, Hash, Layers, Copy, Check } from 'lucide-react';

interface SourceCardProps {
  source: Source;
  index: number;
  isSelected?: boolean;
  onSelect?: () => void;
  additionalContent?: string;
  distance?: number;
  rerankScore?: number | null;
  onInspectDocument?: (docId: string) => void;
}

export const SourceCard: React.FC<SourceCardProps> = ({
  source,
  index,
  isSelected,
  onSelect,
  additionalContent,
  distance,
  rerankScore,
  onInspectDocument,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const toggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    setExpanded(!expanded);
    if (onSelect) onSelect();
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const pagesText = source.page_numbers?.length
    ? source.page_numbers.join(', ')
    : 'N/A';

  return (
    <div
      className={`source-card ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
      style={{
        borderColor: isSelected ? 'var(--primary-indigo)' : undefined,
        boxShadow: isSelected ? '0 0 14px var(--primary-indigo-glow)' : undefined,
      }}
    >
      <div className="source-card-header">
        <div className="source-badge-index">
          #{index + 1}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="source-path" title={source.section_path ?? undefined}>
            {source.section_path || 'Section / Heading'}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
            <span className="source-pages">
              <BookOpen size={11} /> Page {pagesText}
            </span>
            {distance !== undefined && (
              <span style={{ fontSize: '0.72rem', color: '#06b6d4', fontFamily: 'var(--font-mono)' }}>
                Dist: {distance.toFixed(4)}
              </span>
            )}
            {rerankScore !== undefined && rerankScore !== null && (
              <span style={{ fontSize: '0.72rem', color: '#10b981', fontFamily: 'var(--font-mono)' }}>
                Score: {rerankScore.toFixed(3)}
              </span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          {onInspectDocument && (
            <button
              type="button"
              className="btn-ghost"
              style={{ fontSize: '0.7rem', padding: '0.2rem 0.4rem' }}
              onClick={(e) => {
                e.stopPropagation();
                onInspectDocument(source.document_id);
              }}
              title="Inspect in Document Library"
            >
              <FileText size={12} /> Doc
            </button>
          )}

          <button
            type="button"
            onClick={toggleExpand}
            className="btn-ghost"
            style={{ padding: '0.2rem' }}
            title={expanded ? 'Collapse Metadata' : 'Expand Metadata'}
          >
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {/* Expanded Technical Details Drawer */}
      {expanded && (
        <div className="source-details-drawer">
          <div className="source-detail-item">
            <span className="source-detail-label">
              <Layers size={10} style={{ display: 'inline', marginRight: 2 }} /> Section Path
            </span>
            <span className="source-detail-value" title={source.section_path ?? undefined}>
              {source.section_path || 'Root Document'}
            </span>
          </div>

          <div className="source-detail-item">
            <span className="source-detail-label">
              <BookOpen size={10} style={{ display: 'inline', marginRight: 2 }} /> Page(s)
            </span>
            <span className="source-detail-value">
              {pagesText}
            </span>
          </div>

          <div className="source-detail-item">
            <span className="source-detail-label">
              <FileText size={10} style={{ display: 'inline', marginRight: 2 }} /> Document Version ID
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="source-detail-value" title={source.document_id}>
                {source.document_id || 'N/A'}
              </span>
              <button
                type="button"
                className="btn-ghost"
                style={{ padding: '0.1rem 0.2rem' }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCopy(source.document_id, `doc-${source.chunk_id}`);
                }}
              >
                {copiedId === `doc-${source.chunk_id}` ? <Check size={11} color="#10b981" /> : <Copy size={11} />}
              </button>
            </div>
          </div>

          <div className="source-detail-item">
            <span className="source-detail-label">
              <Hash size={10} style={{ display: 'inline', marginRight: 2 }} /> Chunk ID
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span className="source-detail-value" title={source.chunk_id}>
                {source.chunk_id || 'N/A'}
              </span>
              <button
                type="button"
                className="btn-ghost"
                style={{ padding: '0.1rem 0.2rem' }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCopy(source.chunk_id, `chunk-${source.chunk_id}`);
                }}
              >
                {copiedId === `chunk-${source.chunk_id}` ? <Check size={11} color="#10b981" /> : <Copy size={11} />}
              </button>
            </div>
          </div>

          {additionalContent && (
            <div style={{ gridColumn: '1 / -1', marginTop: '0.5rem' }}>
              <span className="source-detail-label" style={{ marginBottom: '0.2rem', display: 'block' }}>
                Chunk Content Excerpt
              </span>
              <div
                style={{
                  background: 'rgba(0,0,0,0.4)',
                  padding: '0.6rem 0.8rem',
                  borderRadius: 6,
                  color: '#cbd5e1',
                  fontSize: '0.78rem',
                  lineHeight: '1.5',
                  maxHeight: 180,
                  overflowY: 'auto',
                  overflowWrap: 'anywhere',
                  wordBreak: 'break-word',
                }}
              >
                {additionalContent}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
