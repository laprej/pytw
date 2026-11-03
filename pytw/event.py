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
        # (rank, n), assigned when the event is sent to another rank; names it in anti-messages
        self.uid = None
        self._reset()

    def _reset(self):
        """Initialize the bookkeeping that belongs to the rank holding this copy of the event."""
        # dst LP's RNG state prior to the forward handler, restored on rollback
        self.rng_state = None
        # number of remote events sent while processing this event, undone on rollback
        self.remote_sent = 0
        # events sent while processing this event, cancelled on rollback
        self.sent: list[Event] = []
        self.processed = False
        # set when the sender rolls back; cancelled events are skipped, never processed
        self.cancelled = False
        # set once the event has been sent to another rank, which must then be sent an anti-message to cancel it
        self.transmitted = False

    # rank-local bookkeeping, not sent over MPI
    _LOCAL = ("rng_state", "remote_sent", "sent", "processed", "cancelled", "transmitted")

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if k not in self._LOCAL}

    def __setstate__(self, state):
        self._reset()
        vars(self).update(state)

    def __lt__(self, other):
        if other is None:
            return self.timestamp
        return self.timestamp < other.timestamp
