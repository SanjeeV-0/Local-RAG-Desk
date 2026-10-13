/**
 * Document Library: manage the ingested corpus (upload, version history,
 * retry, promote a version to current, delete). One card per LOGICAL
 * document in the list view -- versions only ever appear nested inside a
 * document's detail view, never as their own top-level card. Every
 * action (retry/delete/set-current) calls its endpoint with both
 * `logical_document_id` and `version_id` explicitly, matching the backend's
 * two-identifier contract (see app/api/documents.py).
 */
import React, { useState, useEffect } from 'react';
import {
  DocumentUploadResult,
  DocumentVersion,
  LogicalDocumentSummary,
} from '../types/document';
import {
  listDocuments,
  getDocumentVersions,
  retryDocumentVersion,
  deleteDocumentVersion,
  deleteLogicalDocument,
  setCurrentVersion,
  DocumentApiUnavailableError,
} from '../api/documents';
import { StatusBadge } from './StatusBadge';
import { ConfirmModal } from './ConfirmModal';
import { UploadDocumentModal } from './UploadDocumentModal';
import {
  FileText,
  Search,
  RefreshCw,
  Trash2,
  AlertTriangle,
  History,
  ArrowLeft,
  CheckCircle2,
  Info,
  UploadCloud,
} from 'lucide-react';

interface DocumentWorkspaceProps {
  initialDocumentId?: string | null;
  onNavigateToRetrieval?: (docId: string) => void;
}

export const DocumentWorkspace: React.FC<DocumentWorkspaceProps> = ({
  initialDocumentId,
  onNavigateToRetrieval,
}) => {
  const [documents, setDocuments] = useState<LogicalDocumentSummary[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(initialDocumentId || null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'READY' | 'FAILED' | 'PROCESSING'>('ALL');
  const [isLoading, setIsLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [isApiUnavailable, setIsApiUnavailable] = useState(false);

  // Upload modal
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  // Retry state
  const [retryingVersionId, setRetryingVersionId] = useState<string | null>(null);

  // Set-current state
  const [settingCurrentVersionId, setSettingCurrentVersionId] = useState<string | null>(null);

  // Deletion modals
  const [versionToDelete, setVersionToDelete] = useState<{
    version: DocumentVersion;
    doc: LogicalDocumentSummary;
  } | null>(null);
  const [logicalDocToDelete, setLogicalDocToDelete] = useState<LogicalDocumentSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Version history for the currently selected document. GET /documents only
  // returns each document's current version, so the full history behind the
  // detail view is fetched separately from GET /documents/{id}/versions.
  const [documentVersions, setDocumentVersions] = useState<DocumentVersion[]>([]);
  const [isLoadingVersions, setIsLoadingVersions] = useState(false);
  const [versionsError, setVersionsError] = useState<string | null>(null);

  const loadDocuments = async () => {
    setIsLoading(true);
    setActionError(null);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setIsApiUnavailable(false);
    } catch (err: any) {
      if (err instanceof DocumentApiUnavailableError) {
        setIsApiUnavailable(true);
        setDocuments([]);
      } else {
        setActionError(err?.message || 'Failed to load document library.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    if (initialDocumentId) {
      setSelectedDocId(initialDocumentId);
    }
  }, [initialDocumentId]);

  // A document can be selected either by its own logical_document_id, or by
  // the id of its current version (e.g. when navigating here from a Research
  // / Retrieval result, which only ever references current READY versions).
  const selectedDocument = documents.find(
    (d) =>
      d.logical_document_id === selectedDocId ||
      d.current_version?.id === selectedDocId
  );

  // Fetches (or re-fetches) version history for one logical document.
  // Used by action handlers (retry/delete) to refresh the detail view's
  // version list after a mutation, since the selected document's
  // logical_document_id doesn't change and so wouldn't otherwise re-trigger
  // the auto-load effect below.
  const refreshDocumentVersions = async (logicalDocumentId: string) => {
    setIsLoadingVersions(true);
    setVersionsError(null);
    try {
      const versions = await getDocumentVersions(logicalDocumentId);
      setDocumentVersions(versions);
    } catch (err: any) {
      if (err instanceof DocumentApiUnavailableError) {
        setIsApiUnavailable(true);
      }
      setDocumentVersions([]);
      setVersionsError(err?.message || 'Failed to load version history.');
    } finally {
      setIsLoadingVersions(false);
    }
  };

  useEffect(() => {
    if (!selectedDocument) {
      setDocumentVersions([]);
      setVersionsError(null);
      return;
    }

    let cancelled = false;

    const loadVersions = async () => {
      setIsLoadingVersions(true);
      setVersionsError(null);
      try {
        const versions = await getDocumentVersions(selectedDocument.logical_document_id);
        if (!cancelled) setDocumentVersions(versions);
      } catch (err: any) {
        if (cancelled) return;
        if (err instanceof DocumentApiUnavailableError) {
          setIsApiUnavailable(true);
        }
        setDocumentVersions([]);
        setVersionsError(err?.message || 'Failed to load version history.');
      } finally {
        if (!cancelled) setIsLoadingVersions(false);
      }
    };

    loadVersions();

    return () => {
      cancelled = true;
    };
  }, [selectedDocument?.logical_document_id]);

  const filteredDocuments = documents.filter((doc) => {
    const title = doc.current_version?.title ?? '';
    const matchesSearch =
      title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.logical_document_id.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesStatus =
      statusFilter === 'ALL' || doc.current_version?.status === statusFilter;

    return matchesSearch && matchesStatus;
  });

  const describeError = (err: unknown, fallback: string): string => {
    if (err instanceof DocumentApiUnavailableError) {
      return `${err.message} This action could not be performed.`;
    }
    return (err as any)?.message || fallback;
  };

  // Handle Retry
  const handleRetry = async (logicalDocumentId: string, versionId: string) => {
    setRetryingVersionId(versionId);
    setActionError(null);
    setActionSuccess(null);

    try {
      const version = await retryDocumentVersion(logicalDocumentId, versionId);
      setActionSuccess(`Version ${version.version_number} successfully reprocessed and is now READY (Current).`);
      await loadDocuments();
      await refreshDocumentVersions(logicalDocumentId);
    } catch (err) {
      if (err instanceof DocumentApiUnavailableError) {
        setIsApiUnavailable(true);
      }
      setActionError(describeError(err, 'Retry failed.'));
    } finally {
      setRetryingVersionId(null);
    }
  };

  // Handle Make Current
  const handleSetCurrent = async (logicalDocumentId: string, versionId: string) => {
    setSettingCurrentVersionId(versionId);
    setActionError(null);
    setActionSuccess(null);

    try {
      const version = await setCurrentVersion(logicalDocumentId, versionId);
      setActionSuccess(`Version ${version.version_number} is now the current version.`);
      await loadDocuments();
      await refreshDocumentVersions(logicalDocumentId);
    } catch (err) {
      if (err instanceof DocumentApiUnavailableError) {
        setIsApiUnavailable(true);
      }
      setActionError(describeError(err, 'Failed to set version as current.'));
    } finally {
      setSettingCurrentVersionId(null);
    }
  };

  // Handle Delete Version
  const handleConfirmDeleteVersion = async () => {
    if (!versionToDelete) return;
    setIsDeleting(true);
    setActionError(null);

    const logicalDocumentId = versionToDelete.doc.logical_document_id;

    try {
      await deleteDocumentVersion(logicalDocumentId, versionToDelete.version.id);
      setActionSuccess(`Version ${versionToDelete.version.version_number} deleted.`);
      setVersionToDelete(null);
      await loadDocuments();
      await refreshDocumentVersions(logicalDocumentId);
    } catch (err) {
      if (err instanceof DocumentApiUnavailableError) {
        setIsApiUnavailable(true);
      }
      setActionError(describeError(err, 'Failed to delete version.'));
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle Delete Logical Document
  const handleConfirmDeleteLogical = async () => {
    if (!logicalDocToDelete) return;
    setIsDeleting(true);
    setActionError(null);

    try {
      await deleteLogicalDocument(logicalDocToDelete.logical_document_id);
      setActionSuccess(
        `Logical document "${logicalDocToDelete.current_version?.title ?? logicalDocToDelete.logical_document_id}" and all its versions were removed.`
      );
      setLogicalDocToDelete(null);
      setSelectedDocId(null);
      await loadDocuments();
    } catch (err) {
      if (err instanceof DocumentApiUnavailableError) {
        setIsApiUnavailable(true);
      }
      setActionError(describeError(err, 'Failed to delete document.'));
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle a completed upload call -- outcome distinguishes a genuinely new
  // document from a detected global content duplicate (no new document/
  // version/chunks created) from a new revision of an explicitly targeted
  // existing document. The duplicate case is also handled inline by
  // UploadDocumentModal itself (it stays open with an explanation and a
  // "View Existing Document" action), so this handler's job is just to
  // keep the library list in sync and report what happened once the modal
  // is done.
  const handleUploaded = async (result: DocumentUploadResult) => {
    setActionError(null);

    if (result.outcome === 'duplicate') {
      const updated = result.updated_metadata_fields;
      setActionSuccess(
        `Matching content already exists as "${result.version.title}" (version ${result.version.version_number}).` +
          (updated.length > 0 ? ` Updated: ${updated.join(', ')}.` : '')
      );
    } else if (result.outcome === 'new_version') {
      setActionSuccess(
        `"${result.version.title}" uploaded as version ${result.version.version_number} and submitted for ingestion.`
      );
    } else {
      setActionSuccess(
        `"${result.version.title}" uploaded and submitted for ingestion (version ${result.version.version_number}).`
      );
    }

    await loadDocuments();
  };

  const handleViewExisting = (logicalDocumentId: string) => {
    setSelectedDocId(logicalDocumentId);
  };

  return (
    <div className="document-workspace">
      {/* Purpose banner */}
      <div
        className="glass-card"
        style={{
          padding: '0.85rem 1.25rem',
          marginBottom: '1.25rem',
          background: 'rgba(30, 41, 59, 0.4)',
          borderLeft: '4px solid var(--accent-cyan)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.85rem',
        }}
      >
        <Info size={18} color="#06b6d4" style={{ flexShrink: 0 }} />
        <div style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>
          <strong>Manage the knowledge corpus:</strong> upload and ingest documents, track
          UPLOADED / PROCESSING / READY / FAILED lifecycle status, manage versions, retry failed
          ingestions, and delete versions or entire documents.
        </div>
      </div>

      {isApiUnavailable && (
        <div className="warning-banner" style={{ marginBottom: '1.25rem' }}>
          <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div className="warning-title">Document Management API Not Exposed</div>
            <div className="warning-desc">
              The backend does not currently expose HTTP routes for document management
              (e.g. GET/POST/DELETE <code>/documents</code>). This page reflects the intended
              UI for that workflow, but no real documents can be listed, uploaded, retried, or
              deleted until those routes are added to the backend.
            </div>
          </div>
        </div>
      )}

      {actionSuccess && (
        <div className="success-banner" style={{ marginBottom: '1.25rem' }}>
          <CheckCircle2 size={18} color="#10b981" />
          <span>{actionSuccess}</span>
          <button type="button" className="modal-close-btn" onClick={() => setActionSuccess(null)}>
            &times;
          </button>
        </div>
      )}

      {actionError && (
        <div className="error-banner" style={{ marginBottom: '1.25rem' }}>
          <AlertTriangle size={18} color="#f43f5e" />
          <div style={{ flex: 1 }}>
            <div className="error-title">Action Failed</div>
            <div className="error-desc">{actionError}</div>
          </div>
          <button type="button" className="modal-close-btn" onClick={() => setActionError(null)}>
            &times;
          </button>
        </div>
      )}

      {selectedDocument ? (
        /* ================= DOCUMENT DETAIL VIEW ================= */
        <div className="document-detail-container">
          <button
            type="button"
            className="btn-ghost"
            onClick={() => setSelectedDocId(null)}
            style={{ marginBottom: '1rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}
          >
            <ArrowLeft size={16} /> Back to Document Library
          </button>

          {/* Document Header Card */}
          <div className="glass-card" style={{ padding: '1.75rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f8fafc' }}>
                    {selectedDocument.current_version?.title ?? 'Untitled Document'}
                  </h2>
                  {selectedDocument.current_version ? (
                    <StatusBadge status={selectedDocument.current_version.status} isCurrent={true} />
                  ) : (
                    <span className="status-pill badge-status-uploaded">No current version</span>
                  )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginTop: '0.6rem', color: '#94a3b8', fontSize: '0.8rem', flexWrap: 'wrap' }}>
                  <span>
                    Type: <strong style={{ color: '#cbd5e1' }}>{selectedDocument.current_version?.document_type ?? 'Unknown'}</strong>
                  </span>
                  <span>
                    Source: <strong style={{ color: '#cbd5e1' }}>{selectedDocument.current_version?.source || 'Standard Ingestion'}</strong>
                  </span>
                  <span>
                    Total Versions:{' '}
                    <strong style={{ color: '#cbd5e1' }}>
                      {isLoadingVersions ? '…' : documentVersions.length}
                    </strong>
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.6rem' }}>
                {onNavigateToRetrieval && (
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => onNavigateToRetrieval(selectedDocument.logical_document_id)}
                    style={{ fontSize: '0.8rem' }}
                  >
                    <Search size={14} /> Search This Doc
                  </button>
                )}
                <button
                  type="button"
                  className="btn-danger"
                  onClick={() => setLogicalDocToDelete(selectedDocument)}
                  style={{ fontSize: '0.8rem' }}
                >
                  <Trash2 size={14} /> Delete Logical Doc
                </button>
              </div>
            </div>

            <div style={{ marginTop: '1.25rem', padding: '0.85rem 1rem', background: 'rgba(15,23,42,0.6)', borderRadius: 8, fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>
              <div>Logical Document ID: <span style={{ color: '#f8fafc' }}>{selectedDocument.logical_document_id}</span></div>
              <div style={{ marginTop: '0.2rem' }}>
                Current Active Version ID: <span style={{ color: '#38bdf8' }}>{selectedDocument.current_version?.id || 'None'}</span>
              </div>
            </div>
          </div>

          {/* Version History Table / List */}
          <div className="glass-card" style={{ padding: '1.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
              <History size={18} color="#818cf8" />
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                Version History & Lifecycle States
              </h3>
            </div>

            {isLoadingVersions && (
              <div style={{ padding: '1.5rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.85rem' }}>
                <span className="spinner-ring" style={{ width: 16, height: 16, borderWidth: 2, marginRight: '0.5rem' }} />
                Loading version history...
              </div>
            )}

            {!isLoadingVersions && versionsError && (
              <div className="error-banner" style={{ marginBottom: 0 }}>
                <AlertTriangle size={18} color="#f43f5e" />
                <div style={{ flex: 1 }}>
                  <div className="error-title">Could Not Load Version History</div>
                  <div className="error-desc">{versionsError}</div>
                </div>
              </div>
            )}

            {!isLoadingVersions && !versionsError && documentVersions.length === 0 && (
              <div style={{ padding: '1.5rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.85rem' }}>
                No versions found for this document.
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {!isLoadingVersions && !versionsError && documentVersions.map((version) => {
                const isRetrying = retryingVersionId === version.id;
                const isSettingCurrent = settingCurrentVersionId === version.id;

                return (
                  <div
                    key={version.id}
                    className="version-card glass-card"
                    style={{
                      padding: '1.25rem',
                      borderColor: version.is_current ? 'rgba(99,102,241,0.4)' : undefined,
                      background: version.is_current ? 'rgba(99,102,241,0.06)' : undefined,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <span
                          style={{
                            fontSize: '0.9rem',
                            fontWeight: 700,
                            padding: '0.25rem 0.6rem',
                            borderRadius: 6,
                            background: 'rgba(15,23,42,0.8)',
                            color: '#cbd5e1',
                            fontFamily: 'var(--font-mono)',
                          }}
                        >
                          v{version.version_number}
                        </span>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                            <StatusBadge status={version.status} isCurrent={version.is_current} size="sm" />
                            {version.is_current && (
                              <span style={{ fontSize: '0.75rem', color: '#818cf8', fontWeight: 600 }}>
                                (Active for vector retrieval)
                              </span>
                            )}
                          </div>
                          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.35rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                            <span>Attempt: {version.processing_attempt}</span>
                            <span>Created: {new Date(version.created_at).toLocaleString()}</span>
                          </div>
                        </div>
                      </div>

                      {/* Version Action buttons */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        {version.is_current ? (
                          <span
                            style={{
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              color: '#818cf8',
                              padding: '0.35rem 0.75rem',
                              border: '1px solid rgba(129,140,248,0.4)',
                              borderRadius: 6,
                            }}
                          >
                            Current
                          </span>
                        ) : version.status === 'READY' ? (
                          <button
                            type="button"
                            className="btn-secondary"
                            style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
                            onClick={() =>
                              handleSetCurrent(selectedDocument.logical_document_id, version.id)
                            }
                            disabled={isSettingCurrent}
                          >
                            {isSettingCurrent ? (
                              <>
                                <span className="spinner-ring" style={{ width: 12, height: 12, borderWidth: 2 }} />
                                Setting...
                              </>
                            ) : (
                              <>
                                <CheckCircle2 size={13} />
                                Make Current
                              </>
                            )}
                          </button>
                        ) : null}

                        {version.status === 'FAILED' && (
                          <button
                            type="button"
                            className="btn-primary"
                            style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', background: '#e11d48' }}
                            onClick={() => handleRetry(selectedDocument.logical_document_id, version.id)}
                            disabled={isRetrying}
                          >
                            {isRetrying ? (
                              <>
                                <span className="spinner-ring" style={{ width: 12, height: 12, borderWidth: 2 }} />
                                Retrying...
                              </>
                            ) : (
                              <>
                                <RefreshCw size={13} />
                                Retry Version
                              </>
                            )}
                          </button>
                        )}

                        <button
                          type="button"
                          className="btn-ghost"
                          style={{ fontSize: '0.75rem', color: '#f43f5e' }}
                          onClick={() => setVersionToDelete({ version, doc: selectedDocument })}
                        >
                          <Trash2 size={13} /> Delete
                        </button>
                      </div>
                    </div>

                    {/* Failure reason if FAILED */}
                    {version.last_error && (
                      <div
                        style={{
                          marginTop: '0.75rem',
                          padding: '0.75rem 1rem',
                          background: 'rgba(244,63,94,0.1)',
                          border: '1px solid rgba(244,63,94,0.25)',
                          borderRadius: 6,
                          fontSize: '0.78rem',
                          color: '#fecdd3',
                        }}
                      >
                        <strong>Failure Reason:</strong> {version.last_error}
                      </div>
                    )}

                    {/* Technical version metadata drawer */}
                    <div
                      style={{
                        marginTop: '0.75rem',
                        fontSize: '0.72rem',
                        color: '#64748b',
                        fontFamily: 'var(--font-mono)',
                        overflowWrap: 'anywhere',
                      }}
                    >
                      Version ID: {version.id}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        /* ================= DOCUMENT LIBRARY LIST VIEW ================= */
        <div>
          {/* Page heading + primary upload action */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem',
              flexWrap: 'wrap',
              marginBottom: '1.25rem',
            }}
          >
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                Document Library
              </h2>
              <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                Manage the knowledge corpus ingested into the research agent.
              </p>
            </div>
            <button
              type="button"
              className="btn-primary"
              onClick={() => setIsUploadOpen(true)}
            >
              <UploadCloud size={16} />
              Upload / Ingest Document
            </button>
          </div>

          {/* Controls Bar */}
          <div
            className="glass-card"
            style={{
              padding: '1.25rem 1.5rem',
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1, minWidth: 260 }}>
              <div className="input-wrapper" style={{ flex: 1 }}>
                <input
                  type="text"
                  placeholder="Search documents by title, filename, or ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(15,23,42,0.8)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 8,
                    padding: '0.5rem 0.85rem',
                    color: '#f8fafc',
                    fontSize: '0.85rem',
                  }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ display: 'flex', gap: '0.35rem', background: 'rgba(15,23,42,0.6)', padding: '0.2rem', borderRadius: 8 }}>
                {(['ALL', 'READY', 'FAILED', 'PROCESSING'] as const).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    className={`nav-tab-btn ${statusFilter === filter ? 'active' : ''}`}
                    onClick={() => setStatusFilter(filter)}
                    style={{ fontSize: '0.75rem', padding: '0.3rem 0.65rem' }}
                  >
                    {filter}
                  </button>
                ))}
              </div>

              <button
                type="button"
                className="btn-secondary"
                onClick={loadDocuments}
                disabled={isLoading}
                style={{ fontSize: '0.8rem' }}
              >
                <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
                Refresh
              </button>
            </div>
          </div>

          {/* Document Cards Grid */}
          <div className="documents-grid">
            {filteredDocuments.map((doc) => {
              const current = doc.current_version;
              return (
              <div
                key={doc.logical_document_id}
                className="glass-card doc-card"
                style={{
                  padding: '1.5rem',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedDocId(doc.logical_document_id)}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '0.75rem', marginBottom: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <FileText size={18} color="#06b6d4" />
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 600 }}>
                        {current?.document_type ?? 'Unknown type'}
                      </span>
                    </div>
                    {current ? (
                      <StatusBadge status={current.status} isCurrent={true} size="sm" />
                    ) : (
                      <span className="status-pill badge-status-uploaded status-pill-sm">No current version</span>
                    )}
                  </div>

                  <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.5rem', lineHeight: '1.4' }}>
                    {current?.title ?? 'Untitled document'}
                  </h3>

                  <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '1rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <div>Source: <span style={{ color: '#cbd5e1' }}>{current?.source || 'Local Ingestion'}</span></div>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b' }}>
                  <span>
                    {current ? <>Current <strong>v{current.version_number}</strong></> : 'No current version'}
                  </span>
                  <span style={{ color: '#818cf8', fontWeight: 600 }}>
                    Manage Versions &rarr;
                  </span>
                </div>
              </div>
              );
            })}
          </div>

          {isLoading && documents.length === 0 && (
            <div className="glass-card empty-state" style={{ padding: '3rem' }}>
              <span className="spinner-ring" style={{ width: 28, height: 28, borderWidth: 3, margin: '0 auto 1rem auto', display: 'block' }} />
              <p className="empty-title">Loading Document Library...</p>
            </div>
          )}

          {!isLoading && filteredDocuments.length === 0 && (
            <div className="glass-card empty-state" style={{ padding: '3rem' }}>
              <FileText className="empty-icon" />
              <p className="empty-title">
                {isApiUnavailable ? 'Document Library Unavailable' : 'No Documents Found'}
              </p>
              <p className="empty-desc">
                {isApiUnavailable
                  ? 'The backend does not currently expose a document listing endpoint, so no documents can be shown.'
                  : searchQuery || statusFilter !== 'ALL'
                  ? 'No documents match the active filter criteria.'
                  : 'No ingested documents currently in the library. Use "Upload / Ingest Document" to add one.'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Confirmation Dialog: Delete Version */}
      <ConfirmModal
        isOpen={!!versionToDelete}
        title="Delete Document Version"
        description={`Are you sure you want to delete version ${versionToDelete?.version.version_number} of "${versionToDelete?.doc.current_version?.title ?? 'this document'}"?`}
        details={
          <div>
            <p><strong>Consequences:</strong></p>
            <ul style={{ paddingLeft: '1.25rem', marginTop: '0.35rem', lineHeight: '1.5' }}>
              <li>The physical file and associated vector chunks will be deleted.</li>
              {versionToDelete?.version.is_current && (
                <li style={{ color: '#38bdf8' }}>
                  This is the <strong>current version</strong>. Deleting it will automatically promote the newest remaining <strong>READY</strong> version to current.
                </li>
              )}
            </ul>
          </div>
        }
        confirmLabel="Delete Version"
        isDestructive={true}
        isLoading={isDeleting}
        onConfirm={handleConfirmDeleteVersion}
        onCancel={() => setVersionToDelete(null)}
      />

      {/* Confirmation Dialog: Delete Logical Document */}
      <ConfirmModal
        isOpen={!!logicalDocToDelete}
        title="Delete Entire Logical Document"
        description={`Are you sure you want to delete the entire logical document "${logicalDocToDelete?.current_version?.title ?? 'this document'}"?`}
        details={
          <div style={{ color: '#fda4af' }}>
            <p><strong>CRITICAL WARNING:</strong></p>
            <ul style={{ paddingLeft: '1.25rem', marginTop: '0.35rem', lineHeight: '1.5' }}>
              <li>All {documentVersions.length} version(s) of this document will be permanently removed.</li>
              <li>All extracted pages, sections, and vector embeddings in PostgreSQL + pgvector will be purged.</li>
              <li>All physical files stored in storage will be deleted.</li>
            </ul>
          </div>
        }
        confirmLabel="Delete Entire Document"
        isDestructive={true}
        isLoading={isDeleting}
        onConfirm={handleConfirmDeleteLogical}
        onCancel={() => setLogicalDocToDelete(null)}
      />

      {/* Upload / Ingest Modal */}
      <UploadDocumentModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploaded={handleUploaded}
        onViewExisting={handleViewExisting}
        existingDocuments={documents}
      />
    </div>
  );
};
