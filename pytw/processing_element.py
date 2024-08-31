from pytw.engine import Engine
from pytw.event import Event
from pytw.logical_process import LogicalProcess as LP
from pytw.mpi import MPIBase


class ProcessingElement(MPIBase):
    """Processing Element (PE)

    PEs can be thought of as a CPU containing multiple Logical Processes (LPs).
    """

    def __init__(self, e: Engine) -> None:
        super().__init__()
        self.engine = e
        self.lp: list[LP] = []

    def next_event(self) -> Event:
        min = self.lp[0].peek()
        mindex = 0
        # loop over all LPs and find the lowest timestamp event
        for i in range(len(self.lp)):
            ts = self.lp[i].peek()
            if ts is None:
                continue
            if ts < min:
                mindex = i
                min = self.lp[i].peek()

        return self.lp[mindex].next_event()
