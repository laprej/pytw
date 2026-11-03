import itertools
from queue import Queue

from mpi4py import MPI

from pytw.engine import Engine
from pytw.event import Event
from pytw.mpi import MPIBase

# events and anti-messages share a tag so MPI's non-overtaking guarantee keeps an
# anti-message from arriving before the event it cancels
EVENT_TAG = 1


class ProcessingElement(MPIBase):
    """Processing Element (PE)

    PEs can be thought of as a CPU containing multiple Logical Processes (LPs).
    There is one PE per MPI rank; LP i lives on rank i % size.
    """

    def __init__(self, e: Engine) -> None:
        super().__init__()
        self.engine = e
        self.lp: list = []
        self.lp_map: dict = {}
        self.buffer = Queue()
        # events received from other ranks that may still be cancelled, by uid
        self.remote_events: dict = {}
        self.uids = itertools.count()
        # messages (events and anti-messages) sent to and received from other ranks
        self.msgs_sent = 0
        self.msgs_recvd = 0
        self.requests: list = []

    def rank_of(self, lp_id) -> int:
        return 0 if self.size == 1 else lp_id % self.size

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
        """Move buffered events into their destination LP's event list, or send them to its rank."""
        while not self.buffer.empty():
            e = self.buffer.get()
            if e.cancelled:
                continue
            if e.dst_lp in self.lp_map:
                self.lp_map[e.dst_lp].receive(e)
            else:
                e.uid = (self.rank, next(self.uids))
                e.transmitted = True
                self._send(e, self.rank_of(e.dst_lp))

    def cancel(self, e: Event) -> None:
        """Cancel an event sent by one of this PE's LPs."""
        if e.dst_lp in self.lp_map:
            self.lp_map[e.dst_lp].cancel(e)
        elif e.transmitted:
            # an anti-message: the uid alone tells the other rank which event to cancel
            self._send(e.uid, self.rank_of(e.dst_lp))
        else:
            # still in the buffer, so deliver() will skip it
            e.cancelled = True

    def _send(self, msg, rank) -> None:
        self.requests.append(self.comm.isend(msg, dest=rank, tag=EVENT_TAG))
        self.msgs_sent += 1

    def poll(self) -> None:
        """Receive every event and anti-message that has arrived from other ranks."""
        status = MPI.Status()
        while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=EVENT_TAG, status=status):
            msg = self.comm.recv(source=status.Get_source(), tag=EVENT_TAG)
            self.msgs_recvd += 1
            if isinstance(msg, Event):
                self.remote_events[msg.uid] = msg
                self.lp_map[msg.dst_lp].receive(msg)
            else:
                e = self.remote_events.pop(msg)
                self.lp_map[e.dst_lp].cancel(e)

    def complete_sends(self) -> None:
        """Release finished send requests; call once every message has been received."""
        MPI.Request.waitall(self.requests)
        self.requests = []
