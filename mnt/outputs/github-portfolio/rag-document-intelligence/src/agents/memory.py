"""Shared sliding-window memory for multi-agent conversations."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List


@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


class AgentMemory:
    """Rolling conversation buffer shared across all agents in the graph."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._buffer: Deque[Message] = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str) -> None:
        self._buffer.append(Message(role=role, content=content))

    def get_history(self) -> str:
        if not self._buffer:
            return ""
        return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in self._buffer)

    def get_messages(self) -> List[Message]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
