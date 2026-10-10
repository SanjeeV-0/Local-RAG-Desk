import asyncio
import sys
from pathlib import Path

from sqlalchemy import delete

# Tracked-row state lives on the `Engine` object itself (`Engine.info`, a
# dict SQLAlchemy provides for exactly this kind of attached bookkeeping),
# NOT as plain module-level globals. This file gets imported under two
# different qualified names in this project (pytest's own conftest loader
# imports it as top-level `conftest`, while `tests/integration/test_session_
# cleanup.py` imports it as `tests.conftest` -- with no `tests/__init__.py`,
# Python treats these as two separate module objects with two separate sets
# of globals). Module-level globals would silently split into two
# disconnected copies across that boundary -- the event hook would update
# one copy, and a test calling `cleanup_document_test_data()` directly would
# read the other, empty, copy (confirmed by direct reproduction while
# building this). The one object guaranteed to be the SAME singleton
# regardless of which import path reaches it is `app.core.database.engine`
# itself, so tracked IDs are attached there instead.
#
# Record of rows this pytest PROCESS actually caused to be inserted into the
# real configured database -- populated exclusively by `_capture_insert`
# (an engine-level event hook registered in `pytest_sessionstart`, below).
# This is the ONLY source of "ownership" `cleanup_document_test_data` ever
# consults: a row's primary key lands here if and only if an INSERT against
# its table was actually observed executing successfully during this run,
# never because a row merely exists in a table. Pre-existing rows (inserted
# by a prior process, e.g. a developer's own uploads) can never appear here,
# and can therefore never be deleted by this mechanism.
_TRACKED_DOCUMENT_IDS_KEY = "cleanup_tracked_document_ids"
_TRACKED_STORED_FILE_IDS_KEY = "cleanup_tracked_stored_file_ids"
_TRACKED_BLOB_HASHES_KEY = "cleanup_tracked_blob_hashes"
_TRACKED_STORAGE_KEYS_KEY = "cleanup_tracked_storage_keys"


def _tracked_sets(engine):
    """Return (and lazily initialize) this engine's four tracked-id sets.

    Attached directly as a plain attribute on the sync `Engine` object
    (SQLAlchemy's `Engine` does not expose a built-in `.info` dict the way
    `Connection` does) -- `Engine` instances allow arbitrary attributes, and
    this is the one object guaranteed to be the SAME singleton regardless of
    which duplicate module instance (see the top-of-file comment) is asking.
    """

    sync_engine = engine.sync_engine

    if not hasattr(sync_engine, "_cleanup_tracking"):
        sync_engine._cleanup_tracking = {
            _TRACKED_DOCUMENT_IDS_KEY: set(),
            _TRACKED_STORED_FILE_IDS_KEY: set(),
            _TRACKED_BLOB_HASHES_KEY: set(),
            _TRACKED_STORAGE_KEYS_KEY: set(),
        }

    tracking = sync_engine._cleanup_tracking

    return (
        tracking[_TRACKED_DOCUMENT_IDS_KEY],
        tracking[_TRACKED_STORED_FILE_IDS_KEY],
        tracking[_TRACKED_BLOB_HASHES_KEY],
        tracking[_TRACKED_STORAGE_KEYS_KEY],
    )


def _make_capture_insert_hook(engine):
    """Build the `after_cursor_execute` callback bound to one specific
    engine's tracked-id sets (see `_tracked_sets`)."""

    tracked_document_ids, tracked_stored_file_ids, tracked_blob_hashes, tracked_storage_keys = (
        _tracked_sets(engine)
    )

    def _capture_insert(conn, cursor, statement, parameters, context, executemany) -> None:
        """Record real, successful inserts into the three tables this
        fixture is responsible for cleaning up.

        This fires once per executed statement, AFTER it has already
        succeeded against the real database -- so for `documents` and
        `stored_files` (neither ever uses `ON CONFLICT`), being called at
        all means a row was genuinely inserted. `file_blobs` is the one
        exception: its only insert path (`FileBlobRepository.
        try_create_pending`) uses `ON CONFLICT (content_hash) DO NOTHING`,
        which still "executes" on a conflict without inserting anything --
        `cursor.rowcount` distinguishes a real insert (1) from a no-op
        conflict (0), so a losing racer in a concurrency test is never
        mistaken for having created the row the winner already owns.

        `parameters` is a single dict for one row, or a list of dicts for
        an executemany batch -- both are handled uniformly below.
        """

        stripped = statement.lstrip()
        param_batches = parameters if isinstance(parameters, list) else [parameters]

        if stripped.startswith("INSERT INTO documents"):
            for params in param_batches:
                document_id = params.get("id") if params else None
                if document_id is not None:
                    tracked_document_ids.add(document_id)

        elif stripped.startswith("INSERT INTO stored_files"):
            for params in param_batches:
                if not params:
                    continue
                stored_file_id = params.get("id")
                if stored_file_id is not None:
                    tracked_stored_file_ids.add(stored_file_id)
                storage_key = params.get("storage_key")
                if storage_key:
                    tracked_storage_keys.add(storage_key)

        elif stripped.startswith("INSERT INTO file_blobs"):
            if getattr(cursor, "rowcount", 0) != 1:
                # ON CONFLICT DO NOTHING produced no row -- this process did
                # not create it, so it must not be tracked (and therefore
                # never deleted) by this run.
                return
            for params in param_batches:
                if not params:
                    continue
                content_hash = params.get("content_hash")
                if content_hash:
                    tracked_blob_hashes.add(content_hash)
                storage_key = params.get("storage_key")
                if storage_key:
                    tracked_storage_keys.add(storage_key)

    return _capture_insert


def pytest_sessionstart() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Imported here (not at module level) to preserve this conftest's
    # existing convention of never importing the application/database stack
    # as a side effect of collection -- only once pytest has actually
    # started a session.
    from sqlalchemy import event

    from app.core.database import engine

    event.listen(engine.sync_engine, "after_cursor_execute", _make_capture_insert_hook(engine))


async def cleanup_document_test_data() -> None:
    """Remove ONLY the document-lifecycle rows and physical files this
    pytest process is positively known to have created (see
    `_make_capture_insert_hook`), leaving every pre-existing row and file
    untouched.

    This replaces an earlier version of this function that issued an
    unconditional `DELETE` against every document-lifecycle table (and
    wiped the entire storage root) -- that blanket approach deleted
    development data that happened to already be in the configured database
    before the test session ran, because it could not distinguish
    "created by this run" from "already there". This version can only ever
    delete a primary key it watched get inserted.

    Deletes children before parents in FK-safe order: `StoredFile` rows
    first (no `ON DELETE CASCADE` from `Document`), then `Document` rows
    (which cascade to `ChunkPageMap`/`DocumentChunk`/`DocumentSection`/
    `DocumentPage` automatically), then `FileBlob` rows -- and a blob is
    only ever deleted if NO `StoredFile` (tracked or not) still references
    it, so a blob shared with an untracked/pre-existing reference is never
    removed. If nothing was tracked, every step below matches zero rows and
    this function does nothing -- it is always safe to call, and idempotent
    if called more than once.
    """

    from sqlalchemy import select

    from app.core.database import async_session_factory, engine
    from app.core.models import Document, FileBlob, StoredFile

    tracked_document_ids, tracked_stored_file_ids, tracked_blob_hashes, tracked_storage_keys = (
        _tracked_sets(engine)
    )

    deleted_storage_keys: set[str] = set()

    async with async_session_factory() as session:
        if tracked_stored_file_ids:
            await session.execute(
                delete(StoredFile).where(StoredFile.id.in_(tracked_stored_file_ids))
            )

        if tracked_document_ids:
            await session.execute(delete(Document).where(Document.id.in_(tracked_document_ids)))

        if tracked_blob_hashes:
            # Only ever delete a blob that is BOTH one we tracked AND
            # currently unreferenced by any StoredFile row, tracked or not
            # -- the same reference-counting guarantee production code
            # applies in `DocumentDeletionService`.
            still_referenced = (
                select(StoredFile.content_hash)
                .where(StoredFile.content_hash == FileBlob.content_hash)
                .exists()
            )

            deletable_result = await session.execute(
                select(FileBlob.content_hash, FileBlob.storage_key).where(
                    FileBlob.content_hash.in_(tracked_blob_hashes),
                    ~still_referenced,
                )
            )
            deletable_rows = deletable_result.all()

            if deletable_rows:
                deletable_hashes = [row.content_hash for row in deletable_rows]
                deleted_storage_keys.update(row.storage_key for row in deletable_rows)

                await session.execute(
                    delete(FileBlob).where(FileBlob.content_hash.in_(deletable_hashes))
                )

        await session.commit()

    _cleanup_tracked_files(tracked_storage_keys, deleted_storage_keys)

    # Processed rows are done; drop them so a later call in the same process
    # (e.g. the explicit unit tests in test_session_cleanup.py, followed by
    # the real end-of-session call) re-queries empty sets instead of
    # harmlessly re-matching zero rows forever -- purely a memory/clarity
    # tidy-up, not a correctness requirement (re-matching zero rows is
    # already a safe no-op).
    tracked_document_ids.clear()
    tracked_stored_file_ids.clear()
    tracked_blob_hashes.clear()
    tracked_storage_keys.clear()


def _cleanup_tracked_files(
    tracked_storage_keys: set[str],
    additionally_confirmed_keys: set[str],
) -> None:
    """Remove physical files at exactly the tracked storage keys that were
    just confirmed safe to delete at the DB level (unreferenced blobs) --
    never anything else on disk, and never a directory-wide sweep.

    A `StoredFile`'s own `storage_key` is a denormalized copy of its blob's
    key (see `app.core.models.StoredFile`); if that blob is still referenced
    by anything, its file must not be deleted, so only keys confirmed
    deletable above are removed, even though this process tracked every
    `StoredFile`'s key too.
    """

    from app.config.settings import get_settings

    storage_root = Path(get_settings().storage_root)

    if not storage_root.is_dir():
        return

    for storage_key in tracked_storage_keys:
        if storage_key not in additionally_confirmed_keys:
            # This key belongs to a blob that is either still referenced,
            # or was never confirmed deletable above -- leave the file
            # alone.
            continue

        file_path = storage_root / storage_key

        if file_path.is_file():
            file_path.unlink()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Guaranteed end-of-session cleanup, scoped to exactly what this run
    tracked (see `cleanup_document_test_data`).

    pytest calls this hook once after the whole test run finishes --
    regardless of how many tests failed, or whether any assertion raised --
    so rows/files THIS RUN created never carry forward into the next run.
    It can never remove anything this run did not itself create.
    """

    try:
        asyncio.run(cleanup_document_test_data())
    except Exception as exc:
        message = (
            "\n"
            + "=" * 70
            + "\nPOST-TEST DOCUMENT DATA CLEANUP FAILED\n"
            + f"{type(exc).__name__}: {exc}\n"
            + "Rows/files this run created may still be present in the "
            "configured database/storage root.\n" + "=" * 70 + "\n"
        )

        terminal_reporter = session.config.pluginmanager.getplugin("terminalreporter")
        if terminal_reporter is not None:
            terminal_reporter.write_line(message, red=True, bold=True)
        else:
            print(message, file=sys.stderr)
