"""Orchestrates the whole document ingestion pipeline: loading -> global
duplicate-content detection -> shared-blob storage -> structural extraction
-> sectioning -> semantic shredding -> size-guarded chunking -> embedding ->
persistence -> lifecycle status transitions.

This is the single entry point both `POST /documents` (new upload) and
`POST .../retry` (reprocessing a FAILED version) call -- there is exactly
one ingestion code path, not two. It owns transaction boundaries and
failure-recovery (marking a version FAILED with the real exception message)
but delegates every individual step (structure extraction, chunking,
embedding, blob storage) to its own dedicated module.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Document, DocumentPage, DocumentStatus, StoredFile
from app.core.repositories.document import DocumentRepository
from app.core.repositories.stored_file import StoredFileRepository
from app.core.services.document import DocumentService
from app.embeddings.provider import EmbeddingProvider
from app.ingestion.blob_service import BlobService
from app.ingestion.chunk_service import ChunkService
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.resolver import resolve_loader_for_filename
from app.ingestion.schemas import DocumentInput, IngestOutcome
from app.ingestion.section_builder import SectionBuilder
from app.ingestion.section_service import SectionService
from app.ingestion.semantic_shredder import shred_semantically
from app.ingestion.size_guard import apply_size_guard, fragment_table_units
from app.ingestion.structure_extractor import StructureExtractor
from app.storage.interface import FileStorage


@dataclass(frozen=True)
class IngestionResult:
    """What `IngestionService.ingest_file` actually did -- see
    `app.ingestion.schemas.IngestOutcome` for what each outcome means."""

    document: Document
    outcome: IngestOutcome
    updated_metadata_fields: list[str] = field(default_factory=list)


class IngestionService:
    """Orchestrates the document ingestion workflow."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        file_storage: FileStorage,
    ) -> None:
        self.session = session
        self.file_storage = file_storage
        self.repository = DocumentRepository(session)
        self.stored_file_repository = StoredFileRepository(session)
        self.document_service = DocumentService(session)
        self.blob_service = BlobService(session, file_storage=file_storage)
        self.structure_extractor = StructureExtractor()
        self.section_builder = SectionBuilder()
        self.section_service = SectionService(session)
        self.chunk_service = ChunkService(
            session,
            embedding_provider=embedding_provider,
        )
        self.embedding_provider = embedding_provider

    async def ingest_file(
        self,
        path: Path,
        loader: DocumentLoader,
        title: str,
        document_type: str,
        logical_document_id: UUID | None = None,
        source: str | None = None,
        original_filename: str | None = None,
    ) -> IngestionResult:
        """Ingest one uploaded file.

        `logical_document_id` is the sole branch point between the two
        dedup-aware service entry points: omitted, this is "upload as a new
        document" (global duplicate-content detection, server-generated
        logical ID on a true miss -- `DocumentService.
        ingest_as_new_or_duplicate`); supplied, this is "upload a new
        version of THIS existing document" (`DocumentService.
        ingest_as_new_version`), which still checks for a global content
        match before minting a new version. Either way, a matched duplicate
        short-circuits here -- no blob write, no new version/section/chunk/
        embedding, no reprocessing -- see `IngestionResult.outcome`.
        """

        file_bytes = path.read_bytes()
        pages = loader.load(path)

        combined_content = "\n\n".join(page.content for page in pages)

        document_input = DocumentInput(
            title=title,
            source=source,
            document_type=document_type,
            content=combined_content,
        )

        if logical_document_id is None:
            document, match = await self.document_service.ingest_as_new_or_duplicate(
                document_input,
            )
            non_duplicate_outcome = IngestOutcome.CREATED
        else:
            document, match = await self.document_service.ingest_as_new_version(
                logical_document_id,
                document_input,
            )
            non_duplicate_outcome = IngestOutcome.NEW_VERSION

        # Idempotency / dedup short-circuit: a matching normalized-content
        # version already exists (globally, per the outcome above) -- no
        # new blob write, no new StoredFile, no reprocessing. This also
        # covers repeated client retries of the exact same upload: the
        # second attempt always re-resolves to this same existing row
        # rather than creating another logical document, version, or chunk
        # set.
        if match.matched:
            return IngestionResult(
                document=document,
                outcome=IngestOutcome.DUPLICATE,
                updated_metadata_fields=match.updated_metadata_fields,
            )

        return await self._store_and_process_new_version(
            document=document,
            file_bytes=file_bytes,
            pages=pages,
            original_filename=original_filename or path.name,
            outcome=non_duplicate_outcome,
        )

    async def _store_and_process_new_version(
        self,
        document: Document,
        file_bytes: bytes,
        pages: list,
        original_filename: str,
        outcome: IngestOutcome,
    ) -> IngestionResult:
        # Blob storage is fully decoupled from this version's identity:
        # `BlobService` only ever returns a blob once its bytes are
        # durably, atomically written and its row committed READY -- see
        # that module for the concurrency/crash-recovery protocol. Several
        # OTHER versions (this document's own earlier versions, or
        # entirely different logical documents) may already reference --
        # or come to reference later -- the exact same blob; this
        # StoredFile row is only ever a reference to it, never an owner.
        blob = await self.blob_service.get_or_create_ready_blob(file_bytes)

        try:
            stored_file = StoredFile(
                document_id=document.id,
                original_filename=original_filename,
                content_hash=blob.content_hash,
                size_bytes=blob.size_bytes,
                storage_key=blob.storage_key,
            )

            self.session.add(stored_file)

            await self.session.commit()

        except Exception:
            await self.session.rollback()
            # Deliberately do NOT delete the blob here: it is a shared
            # resource that may already be referenced by (or about to be
            # referenced by) another version entirely -- only the
            # reference-counted path in `DocumentDeletionService` may ever
            # delete a blob's physical bytes.
            raise

        document = await self.document_service.mark_processing(document)

        await self.session.commit()

        document = await self._process_document(
            document=document,
            pages=pages,
        )

        return IngestionResult(document=document, outcome=outcome)

    async def retry_document(
        self,
        document_id: UUID,
        loader: DocumentLoader | None = None,
    ) -> Document:
        """Retry a FAILED document version, reusing its existing StoredFile
        and physical file -- no new version or stored file is created.

        `loader` is optional: when not supplied, it is resolved from the
        existing StoredFile's `original_filename` (the same loader the
        original upload would have used), since that filename is already
        known to the service and does not need to come from the caller.
        """

        document = await self.repository.get_by_id(document_id)

        if document is None:
            raise ValueError(f"Document {document_id} not found.")

        if document.status != DocumentStatus.FAILED:
            raise ValueError("Only FAILED documents can be retried.")

        stored_file = await self.stored_file_repository.get_by_document_id(document.id)

        if stored_file is None:
            raise ValueError(f"No stored file found for document {document.id}.")

        if loader is None:
            loader = resolve_loader_for_filename(stored_file.original_filename)

        file_bytes = await self.file_storage.retrieve(stored_file.storage_key)

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(stored_file.original_filename).suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(file_bytes)
                temporary_path = Path(temporary_file.name)

            pages = loader.load(temporary_path)

            document = await self.document_service.mark_processing(document)

            await self.session.commit()

            return await self._process_document(
                document=document,
                pages=pages,
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def _process_document(
        self,
        document: Document,
        pages: list,
    ) -> Document:
        document_id = document.id

        try:
            page_ids: dict[int, UUID] = {}

            for page in pages:
                document_page = DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    content=page.content,
                )

                await self.repository.create_page(document_page)

                page_ids[page.page_number] = document_page.id

            # 0.7 similarity threshold is hard-coded here, not a Settings
            # field -- it controls how aggressively adjacent paragraphs are
            # grouped into one chunk before splitting.
            #
            # The final per-chunk size cap (for both prose and table
            # chunks, which share this one budget) is tokenizer-aware when
            # the configured embedding provider exposes `count_tokens`/
            # `max_embedding_tokens` (currently
            # `SentenceTransformerEmbeddingProvider` -- see its module for
            # what that budget represents and why). A provider without that
            # capability (e.g. `DeterministicEmbeddingProvider` and other
            # test fakes) falls back to the previous hard-coded 500
            # word-estimated cap, completely unchanged, so chunk boundaries
            # for every such caller are unaffected by this fallback path.
            token_counter = getattr(self.embedding_provider, "count_tokens", None)
            max_embedding_tokens = getattr(
                self.embedding_provider,
                "max_embedding_tokens",
                None,
            )

            if token_counter is not None and max_embedding_tokens is not None:
                max_chunk_tokens = max_embedding_tokens
                size_guard_kwargs: dict[str, object] = {
                    "count_tokens": token_counter,
                    "reserve_section_path_tokens": True,
                }
            else:
                max_chunk_tokens = 500
                size_guard_kwargs = {}

            structural_units = self.structure_extractor.extract(pages)

            # Tables must be pre-fragmented to fit max_chunk_tokens BEFORE
            # section building/semantic shredding -- see
            # fragment_table_units's docstring for why this ordering is
            # required (shred_semantically/apply_size_guard cannot split an
            # oversized table themselves). section_path/section_level are
            # preserved unchanged on every resulting fragment, so this step
            # never affects which sections SectionBuilder discovers.
            structural_units = fragment_table_units(
                structural_units,
                max_tokens=max_chunk_tokens,
                **size_guard_kwargs,
            )

            section_nodes = self.section_builder.build(structural_units)

            section_map = await self.section_service.persist_sections(
                document.id,
                section_nodes,
            )

            semantic_units = shred_semantically(
                structural_units,
                embedding_provider=self.embedding_provider,
                threshold=0.7,
            )

            child_chunks = apply_size_guard(
                semantic_units,
                section_map=section_map,
                max_tokens=max_chunk_tokens,
                **size_guard_kwargs,
            )

            await self.chunk_service.persist_chunks(
                document_id=document.id,
                page_ids=page_ids,
                chunks=child_chunks,
            )

            document = await self.document_service.mark_ready(document)

            await self.session.commit()

            return document

        except Exception as exc:
            await self.session.rollback()

            document = await self.repository.get_by_id(document_id)

            if document is None:
                raise RuntimeError("Document disappeared after ingestion failure.") from exc

            await self.document_service.mark_failed(
                document,
                str(exc),
            )

            await self.session.commit()

            # Attach the persisted FAILED version to the exception itself
            # (without changing its type or message) so a caller such as
            # the HTTP layer can report that version back to the client
            # instead of losing it behind a generic error.
            exc.failed_document = document

            raise
