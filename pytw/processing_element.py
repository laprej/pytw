from queue import Queue

from pytw.engine import Engine
from pytw.event import Event
from pytw.mpi import MPIBase


class ProcessingElement(MPIBase):
    """Processing Element (PE)

    PEs can be thought of as a CPU containing multiple Logical Processes (LPs).
    """

    def __init__(self, e: Engine) -> None:
        super().__init__()
        self.engine = e
        self.lp: list = []
        self.lp_map: dict = {}
        self.buffer = Queue()

    def add_lp(self, lp) -> None:
        self.lp.append(lp)
        self.lp_map[lp.lp_id] = lp

    def peek(self):
        """Return the LP holding the lowest timestamp event, or None if all are empty."""
        min_lp = None
        # loop over all LPs and find the lowest timestamp event
        for lp in self.lp:
            e = lp.peek()
            if e is None:
                continue
            if min_lp is None or e < min_lp.peek():
                min_lp = lp
        return min_lp

    def next_event(self) -> Event:
        lp = self.peek()
        if lp is None:
            return None
        return lp.next_event()

    def deliver(self) -> None:
        """Move buffered events into their destination LP's event list."""
        while not self.buffer.empty():
            e = self.buffer.get()
            self.lp_map[e.dst_lp].add_event(e)
