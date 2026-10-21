"""HTTP boundary for the two system-instruction layers (see
`app.generation.system_instruction`) -- combined by
`OpenRouterGenerationProvider.generate` as
`[default] + [user] + [built-in prompt]` ahead of the retrieved-chunk
context, which only ever appears in the user message:

- `/settings/default-system-instruction`: the always-applied baseline,
  meant to be configured once and rarely changed (e.g. a house style).
- `/settings/system-instruction`: the ad hoc layer a user adds and clears
  as needed.

Contains no generation logic itself -- this module only reads/writes the
two persisted instruction texts.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import (
    get_default_system_instruction_store,
    get_system_instruction_store,
)
from app.generation.system_instruction import FileSystemInstructionStore

router = APIRouter(
    prefix="/settings",
    tags=["settings"],
)


class SystemInstructionResponse(BaseModel):
    text: str


class SystemInstructionRequest(BaseModel):
    # A generous but bounded cap -- large enough for a genuine multi-
    # paragraph instruction, small enough to guard against accidentally
    # (or deliberately) bloating every single generation request's prompt
    # with an unbounded amount of text. Empty string is valid -- it clears
    # the instruction back to "none configured".
    text: str = Field(max_length=5000)


@router.get(
    "/system-instruction",
    response_model=SystemInstructionResponse,
)
async def get_system_instruction(
    store: Annotated[
        FileSystemInstructionStore,
        Depends(get_system_instruction_store),
    ],
) -> SystemInstructionResponse:
    """Return the current system instruction, or "" if none is configured."""

    return SystemInstructionResponse(text=store.get())


@router.put(
    "/system-instruction",
    response_model=SystemInstructionResponse,
)
async def update_system_instruction(
    request: SystemInstructionRequest,
    store: Annotated[
        FileSystemInstructionStore,
        Depends(get_system_instruction_store),
    ],
) -> SystemInstructionResponse:
    """Persist a new system instruction, replacing whatever was there
    before. Takes effect on the very next `/research/ask` call -- see
    `app.generation.openrouter.OpenRouterGenerationProvider.generate`,
    which reads this fresh on every call rather than caching it."""

    store.set(request.text)

    return SystemInstructionResponse(text=store.get())


@router.get(
    "/default-system-instruction",
    response_model=SystemInstructionResponse,
)
async def get_default_system_instruction(
    store: Annotated[
        FileSystemInstructionStore,
        Depends(get_default_system_instruction_store),
    ],
) -> SystemInstructionResponse:
    """Return the current default (always-applied) system instruction, or
    "" if none is configured."""

    return SystemInstructionResponse(text=store.get())


@router.put(
    "/default-system-instruction",
    response_model=SystemInstructionResponse,
)
async def update_default_system_instruction(
    request: SystemInstructionRequest,
    store: Annotated[
        FileSystemInstructionStore,
        Depends(get_default_system_instruction_store),
    ],
) -> SystemInstructionResponse:
    """Persist a new default system instruction, replacing whatever was
    there before. Takes effect on the very next `/research/ask` call, same
    as `update_system_instruction` above -- never cached."""

    store.set(request.text)

    return SystemInstructionResponse(text=store.get())
