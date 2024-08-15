from pytw.engine import Engine
from pytw.event import Event
from pytw.mpi import MPIBase


class Sequential(Engine):
    def __init__(self) -> None:
        super().__init__("Sequential")
        self.lp = []

        if self.size > 1:
            # Sequential can only have a single rank
            self.comm.Abort()

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
