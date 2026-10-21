"""A single, persisted, user-editable instruction prepended to the
generation system prompt on every call -- ahead of the model's built-in
instructions, and therefore ahead of the retrieved-chunk context too, since
that context only ever appears in the user message, which a chat model
always reads after the system message.

Read fresh on every `OpenRouterGenerationProvider.generate` call, never
cached at process start -- an edit made through `PUT
/settings/system-instruction` takes effect on the very next request, no
backend restart required.

Stored as a single plain-text file under the application's storage root
rather than a database row: there is exactly one instruction, global to the
whole (single-user) deployment, so a file is simpler than a new table and
migration for one string.
"""

from pathlib import Path
from typing import Protocol


class SystemInstructionProvider(Protocol):
    """Supplies the current custom system instruction, if any."""

    def get(self) -> str:
        """Return the current instruction text, or "" if none is set."""


class FileSystemInstructionStore:
    """The only `SystemInstructionProvider` implementation: a plain text
    file under the application's storage root."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self) -> str:
        """Return the current instruction text, or "" if none has ever
        been set -- including when the file has never been created."""

        if not self.path.exists():
            return ""

        return self.path.read_text(encoding="utf-8").strip()

    def set(self, text: str) -> None:
        """Persist `text` as the current instruction, overwriting whatever
        was there before. Empty/whitespace-only `text` clears it back to
        "no instruction configured" -- the file is still written, but reads
        back as ""."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text.strip(), encoding="utf-8")
