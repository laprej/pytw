from __future__ import annotations

from typing import Optional


class Event:
    def __init__(self, ts: Optional[float] = None, src: Optional[LP] = None, dst: Optional[LP] = None) -> None:
        self.timestamp = ts
        self.src_lp = src
        self.dst_lp = dst

    def __lt__(self, other):
        if other is None:
            return self.timestamp
        return self.timestamp < other.timestamp
