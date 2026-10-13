import React from 'react';
import { Cpu, Terminal, Activity, Database, Sparkles, BookOpen, Layers } from 'lucide-react';
import { BackendReadinessResponse } from '../types/research';

export type WorkspaceTab = 'research' | 'retrieval' | 'documents' | 'trace';

interface HeaderProps {
  activeTab: WorkspaceTab;
  onSelectTab: (tab: WorkspaceTab) => void;
  traceEnabled: boolean;
  onToggleTrace: (enabled: boolean) => void;
  isBackendHealthy: boolean | null;
  readinessInfo: BackendReadinessResponse | null;
  hasTraceData: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  traceEnabled,
  onToggleTrace,
  isBackendHealthy,
  readinessInfo,
  hasTraceData,
}) => {
  const isDbReady = readinessInfo?.status === 'ready' && readinessInfo.checks.database;

  return (
    <header className="header" style={{ flexDirection: 'column', gap: '1rem', padding: '1.25rem 1.75rem' }}>
      {/* Top Brand & System Status Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', flexWrap: 'wrap', gap: '1rem' }}>
        <div className="header-brand">
          <div className="brand-icon">
            <Cpu size={22} />
          </div>
          <div>
            <h1 className="brand-title">Production AI Research Agent</h1>
            <p className="brand-subtitle">
              PostgreSQL + pgvector &middot; OpenRouter LLM &middot; Lifecycle & Evidence Workspace
            </p>
          </div>
        </div>

        <div className="header-controls">
          {/* API Health badge */}
          <div className="badge-status">
            <span className={`status-dot ${isBackendHealthy === false ? 'offline' : ''}`} />
            <Activity size={12} />
            {isBackendHealthy === null
              ? 'Checking API...'
              : isBackendHealthy
              ? 'FastAPI Online'
              : 'API Offline'}
          </div>

          {/* Database Readiness badge */}
          <div
            className="badge-status"
            style={{
              background: isDbReady ? 'rgba(6, 182, 212, 0.1)' : 'rgba(244, 63, 94, 0.1)',
              borderColor: isDbReady ? 'rgba(6, 182, 212, 0.25)' : 'rgba(244, 63, 94, 0.25)',
              color: isDbReady ? '#06b6d4' : '#f43f5e',
            }}
          >
            <Database size={12} />
            {readinessInfo === null
              ? 'Checking DB...'
              : isDbReady
              ? 'pgvector Ready'
              : 'DB Disconnected'}
          </div>

          {/* Trace toggle switch */}
          <div className="trace-toggle-box">
            <span className="trace-toggle-label">
              <Terminal size={14} />
              Trace Mode
            </span>
            <button
              type="button"
              className={`toggle-switch ${traceEnabled ? 'active' : ''}`}
              onClick={() => onToggleTrace(!traceEnabled)}
              aria-pressed={traceEnabled}
              aria-label="Toggle Trace Mode"
              title={traceEnabled ? 'Trace Mode ON (includes full retrieval candidates & context)' : 'Trace Mode OFF'}
            >
              <div className="toggle-knob" />
            </button>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: traceEnabled ? '#818cf8' : '#64748b' }}>
              {traceEnabled ? 'ON' : 'OFF'}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs Bar */}
      <nav
        className="nav-tabs-bar"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          width: '100%',
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: '0.85rem',
          flexWrap: 'wrap',
        }}
        role="tablist"
      >
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'research'}
          className={`nav-tab-btn ${activeTab === 'research' ? 'active' : ''}`}
          onClick={() => onSelectTab('research')}
        >
          <Sparkles size={15} />
          Research & Synthesis
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'retrieval'}
          className={`nav-tab-btn ${activeTab === 'retrieval' ? 'active' : ''}`}
          onClick={() => onSelectTab('retrieval')}
        >
          <Layers size={15} />
          Vector Retrieval Explorer
        </button>

        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'documents'}
          className={`nav-tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
          onClick={() => onSelectTab('documents')}
        >
          <BookOpen size={15} />
          Document Library & Versions
        </button>

        {(traceEnabled || hasTraceData) && (
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'trace'}
            className={`nav-tab-btn ${activeTab === 'trace' ? 'active' : ''}`}
            onClick={() => onSelectTab('trace')}
          >
            <Terminal size={15} />
            Retrieval Debug & Trace
          </button>
        )}
      </nav>
    </header>
  );
};
