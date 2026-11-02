"""The MemorySystem protocol every system implements.

Purpose: one interface, so adding a system is a new file plus a config entry (D-013).
Called by: pipeline.py, which calls build() then retrieve() for each question.
Assumes: retrieve() only reads what build() prepared; neither writes run records.
"""

from typing import Protocol

from memtax.schema import Question, Retrieved


class MemorySystem(Protocol):
    async def build(self, q: Question) -> None: ...

    async def retrieve(self, q: Question) -> Retrieved: ...
