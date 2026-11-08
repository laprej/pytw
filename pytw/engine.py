import math
from abc import ABC

from mpi4py import MPI

from pytw.mpi import MPIBase


class Engine(MPIBase, ABC):
    """Engine

    There can only be a single Engine for the entire simulation.  This will determine
    how the simulation behaves: sequential, conversative, or optimistic.
    Engines need to be aware of the MPI details.

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, et) -> None:
        from pytw.logical_process import LogicalProcess as LP

        super().__init__()
        self.lp: list[LP] = []
        self.engine_type = et
        if self.engine_type not in ["Sequential", "Conservative", "Optimistic"]:
            raise RuntimeError("bad engine type")

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

        lp = pe.peek()
        lvt = math.inf if lp is None else lp.peek().timestamp
        return self.comm.allreduce(lvt, op=MPI.MIN)
