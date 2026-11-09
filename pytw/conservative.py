from mpi4py import MPI

from pytw.engine import Engine


class Conservative(Engine):
    """Synchronous conservative engine (YAWNS-style windows), on one or more MPI ranks.

    Like ROSS's conservative mode, it relies on lookahead: every event an LP sends
    to another LP is at least `lookahead` later than the sending LP's current time.
    Each window starts by computing the lowest timestamp of any unprocessed event
    anywhere (see Engine.gvt), with nothing in transit.  No event that will ever
    arrive can then be earlier than that plus lookahead, so every rank processes
    its events below that bound in timestamp order, knowing nothing can arrive in
    its past, and only then exchanges events.  Nothing is ever rolled back, so
    models need no reverse handler.

    The lookahead is the smallest any of the models declares (Model.lookahead) and
    must be positive.  Sends to another LP that break it raise an error, wherever
    the destination LP is, so a model that only works on one rank fails there too.
    """

    def __init__(self) -> None:
        super().__init__("Conservative")
        self.lookahead = None
        # number of windows run, i.e. synchronizations between ranks
        self.windows = 0

    def run(self, pe, end_time: float) -> int:
        """Run until no events remain below end_time.

        Returns:
            int: The number of events processed across all ranks.
        """
        # events sent during model initialization are exempt from the lookahead check:
        # all of them are received before the first window starts
        pe.deliver()
        local = min((lp.model.lookahead for lp in pe.lp), default=float("inf"))
        self.lookahead = self.comm.allreduce(local, op=MPI.MIN)
        if not self.lookahead > 0:
            raise ValueError(f"the conservative engine needs a positive lookahead, not {self.lookahead}")
        pe.lookahead = self.lookahead

        try:
            while (lbts := self.gvt(pe)) < end_time:
                self.windows += 1
                window_end = min(lbts + self.lookahead, end_time)
                while (lp := pe.peek()) is not None and lp.peek().timestamp < window_end:
                    e = lp.next_event()
                    lp.now = e.timestamp
                    # call the handler directly: nothing is rolled back, so skip lp.forward()'s state saving
                    lp.model.event_handler(e, lp.model.state, lp)
                    lp.events_processed += 1
                    pe.deliver()
        finally:
            pe.lookahead = None

        return self.comm.allreduce(sum(lp.events_processed for lp in pe.lp))
