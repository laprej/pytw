import math

from mpi4py import MPI

from pytw.engine import Engine


class Optimistic(Engine):
    """Time Warp, on one or more MPI ranks.

    Rather than processing events in global timestamp order, each rank's LPs take
    turns processing up to `batch` events each, so an LP can run ahead of the
    others and receive stragglers from LPs that are behind it.  Stragglers and
    cancellations roll the LP back (see LogicalProcess.rollback).  GVT is
    recomputed after every `gvt_interval` rounds of turns; processed events older
    than GVT are fossil collected, and the run ends once GVT reaches end_time.
    A longer interval means fewer blocking GVT computations, but more history
    held between them.
    """

    def __init__(self, batch: int = 8, gvt_interval: int = 1) -> None:
        super().__init__("Optimistic")
        self.batch = batch
        self.gvt_interval = gvt_interval
        if batch < 1 or gvt_interval < 1:
            raise ValueError("batch and gvt_interval must be at least 1")
        # most processed events held across this rank's LPs at once, i.e. the rollback window
        self.peak_history = 0

    def gvt(self, pe) -> float:
        """Lowest timestamp of any unprocessed event on any rank.

        Blocking, as in ROSS's MPI_Allreduce GVT: every rank stops processing and
        keeps sending and receiving until every message sent anywhere has been
        received, so none are in transit.  Received stragglers and anti-messages can
        cause rollbacks that send further anti-messages, hence the loop.
        """
        while True:
            pe.deliver()
            pe.poll()
            if self.comm.allreduce(pe.msgs_sent - pe.msgs_recvd) == 0:
                break
        pe.complete_sends()

        lvt = min((e.timestamp for lp in pe.lp if (e := lp.peek()) is not None), default=math.inf)
        return self.comm.allreduce(lvt, op=MPI.MIN)

    def run(self, pe, end_time: float) -> int:
        """Run until GVT reaches end_time.

        Returns:
            int: The number of committed events across all ranks, i.e. processed and
            not rolled back.
        """
        while True:
            gvt = self.gvt(pe)
            self.peak_history = max(self.peak_history, sum(len(lp.processed) for lp in pe.lp))
            for lp in pe.lp:
                lp.fossil_collect(gvt)
            if gvt >= end_time:
                break

            for _ in range(self.gvt_interval):
                for lp in pe.lp:
                    pe.poll()
                    for _ in range(self.batch):
                        e = lp.peek()
                        if e is None or e.timestamp >= end_time:
                            break
                        lp.next_event()
                        lp.now = e.timestamp
                        lp.forward(e)
                        pe.deliver()

        return self.comm.allreduce(sum(lp.events_processed for lp in pe.lp))
