import math

from pytw.engine import Engine


class Optimistic(Engine):
    """Time Warp on a single rank.

    Rather than processing events in global timestamp order, LPs take turns
    processing up to `batch` events each, so an LP can run ahead of the others and
    receive stragglers from LPs that are behind it.  Stragglers and cancellations
    roll the LP back (see LogicalProcess.rollback).  GVT is recomputed after every
    round of turns; processed events older than GVT are fossil collected, and the
    run ends once GVT reaches end_time.
    """

    def __init__(self, batch: int = 8) -> None:
        super().__init__("Optimistic")
        self.batch = batch
        # most processed events held across all LPs at once, i.e. the rollback window
        self.peak_history = 0

        if self.size > 1:
            # multiple ranks need a distributed GVT algorithm
            self.comm.Abort()

    def gvt(self, pe) -> float:
        """Lowest timestamp of any unprocessed event."""
        # on a single rank nothing is in transit once the buffer has been delivered
        pe.deliver()
        return min((e.timestamp for lp in pe.lp if (e := lp.peek()) is not None), default=math.inf)

    def run(self, pe, end_time: float) -> int:
        """Run until GVT reaches end_time.

        Returns:
            int: The number of committed events, i.e. processed and not rolled back.
        """
        while True:
            gvt = self.gvt(pe)
            self.peak_history = max(self.peak_history, sum(len(lp.processed) for lp in pe.lp))
            for lp in pe.lp:
                lp.fossil_collect(gvt)
            if gvt >= end_time:
                break

            for lp in pe.lp:
                for _ in range(self.batch):
                    e = lp.peek()
                    if e is None or e.timestamp >= end_time:
                        break
                    lp.next_event()
                    lp.now = e.timestamp
                    lp.forward(e)
                    pe.deliver()

        return sum(lp.events_processed for lp in pe.lp)
