from __future__ import annotations

import random
from typing import Optional


class Event:
    def __init__(self, ts: float, src=None, dst=None) -> None:
        if ts == None:
            ts = random.random()
        self.timestamp = ts
        self.src_lp = src
        self.dst_lp = dst
        # dst LP's RNG state prior to the forward handler, restored on rollback
        self.rng_state = None
        # number of remote events sent while processing this event, undone on rollback
        self.remote_sent = 0

    def __lt__(self, other):
        if other is None:
            return self.timestamp
        return self.timestamp < other.timestamp
