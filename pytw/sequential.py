from mpi4py import MPI

from pytw.engine import Engine
from pytw.event import Event

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Sequential(Engine):
    def __init__(self) -> None:
        super().__init__()

        if size > 1:
            # Sequential can only have a single rank
            comm.Abort()

    def process_event(self) -> Event:
        min = self.lps[0].peek()
        mindex = 0
        # loop over all LPs and find the lowest timestamp event
        for i in range(len(self.lps)):
            ts = self.lps[i].peek()
            if ts is None:
                continue
            if ts < min:
                mindex = i
                min = self.lps[i].peek()

        return self.lps[mindex].next_event()
