import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DocumentWorkspace } from '../DocumentWorkspace';
import { listDocuments, getDocumentVersions } from '../../api/documents';
import { LogicalDocumentSummary, DocumentVersion } from '../../types/document';

vi.mock('../../api/documents', () => ({
  listDocuments: vi.fn(),
  getDocumentVersions: vi.fn(),
  retryDocumentVersion: vi.fn(),
  deleteDocumentVersion: vi.fn(),
  deleteLogicalDocument: vi.fn(),
  setCurrentVersion: vi.fn(),
  DocumentApiUnavailableError: class DocumentApiUnavailableError extends Error {},
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const CURRENT_VERSION: DocumentVersion = {
  id: 'version-aaa', // the per-VERSION id -- what RetrievalWorkspace's
  // "Inspect Doc" button actually sends (RetrievedChunkResponse.document_id,
  // a foreign key to the version row, never the logical grouping id).
  logical_document_id: 'doc-aaa',
  version_number: 1,
  is_current: true,
  status: 'READY',
  title: 'Vector Search Fundamentals',
  document_type: 'research_paper',
  source: null,
  processing_attempt: 1,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  processing_started_at: null,
  processing_completed_at: null,
  failed_at: null,
  last_error: null,
};

const DOC: LogicalDocumentSummary = {
  logical_document_id: 'doc-aaa',
  current_version: CURRENT_VERSION,
};

describe('DocumentWorkspace Inspect Doc navigation', () => {
  it('resolves a per-version document id (as sent by Inspect Doc) to the correct logical document', async () => {
    // Reproduction attempt for the reported bug: navigate in with the
    // VERSION id, exactly as RetrievalWorkspace's "Inspect Doc" button
    // does (`onSelectDocument(chunk.document_id)`), never the logical id.
    vi.mocked(listDocuments).mockResolvedValue([DOC]);
    vi.mocked(getDocumentVersions).mockResolvedValue([CURRENT_VERSION]);

    render(<DocumentWorkspace initialDocumentId="version-aaa" />);

    // The workspace must resolve the version id to this document's detail
    // view -- not fall through to the "no document selected" library list.
    await waitFor(() =>
      expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument(),
    );
    expect(screen.getByText(/Logical Document ID:/)).toHaveTextContent('doc-aaa');
    expect(screen.queryByText('Document Library')).not.toBeInTheDocument();
  });

  it('also resolves a logical document id directly (the pre-existing, documented contract)', async () => {
    vi.mocked(listDocuments).mockResolvedValue([DOC]);
    vi.mocked(getDocumentVersions).mockResolvedValue([CURRENT_VERSION]);

    render(<DocumentWorkspace initialDocumentId="doc-aaa" />);

    await waitFor(() =>
      expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument(),
    );
  });

  it('falls back to the document library list when the navigation target matches nothing', async () => {
    vi.mocked(listDocuments).mockResolvedValue([DOC]);

    render(<DocumentWorkspace initialDocumentId="some-unrelated-id" />);

    await waitFor(() => expect(listDocuments).toHaveBeenCalled());

    // The library LIST view is shown (not the detail view) -- the document's
    // title may still appear as a card in that list, so the detail-view-only
    // "Logical Document ID:" label is the reliable signal that no document
    // was resolved as "selected".
    expect(screen.getByText('Document Library')).toBeInTheDocument();
    expect(screen.queryByText(/Logical Document ID:/)).not.toBeInTheDocument();
  });

  it('preserves version-selection controls (Make Current) in the resolved detail view', async () => {
    const olderReadyVersion: DocumentVersion = {
      ...CURRENT_VERSION,
      id: 'version-older',
      version_number: 1,
      is_current: false,
    };
    const currentVersion: DocumentVersion = {
      ...CURRENT_VERSION,
      id: 'version-aaa',
      version_number: 2,
      is_current: true,
    };

    vi.mocked(listDocuments).mockResolvedValue([
      { logical_document_id: 'doc-aaa', current_version: currentVersion },
    ]);
    vi.mocked(getDocumentVersions).mockResolvedValue([currentVersion, olderReadyVersion]);

    render(<DocumentWorkspace initialDocumentId="version-aaa" />);

    await waitFor(() =>
      expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument(),
    );

    // The non-current, READY older version still offers "Make Current".
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Make Current/i })).toBeInTheDocument(),
    );
  });

  it('search/status filtering in the library list is unaffected by id-resolution behavior', async () => {
    const otherDoc: LogicalDocumentSummary = {
      logical_document_id: 'doc-bbb',
      current_version: { ...CURRENT_VERSION, id: 'version-bbb', logical_document_id: 'doc-bbb', title: 'Unrelated Report' },
    };

    vi.mocked(listDocuments).mockResolvedValue([DOC, otherDoc]);

    const user = userEvent.setup();
    render(<DocumentWorkspace />);

    await waitFor(() => expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument());
    expect(screen.getByText('Unrelated Report')).toBeInTheDocument();

    const searchInput = screen.getByPlaceholderText(/Search documents by title/i);
    await user.type(searchInput, 'Vector');

    await waitFor(() => expect(screen.queryByText('Unrelated Report')).not.toBeInTheDocument());
    expect(screen.getByText('Vector Search Fundamentals')).toBeInTheDocument();
  });
});
