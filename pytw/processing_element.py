import heapq
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
        # (head timestamp, index into self.lp) per LP, so peek() needn't scan every LP;
        # built on the first peek(), since only the sequential engine uses it
        self.heads = None

    def rank_of(self, lp_id) -> int:
        return 0 if self.size == 1 else lp_id % self.size

    def add_lp(self, lp) -> None:
        lp.pe_index = len(self.lp)
        self.lp.append(lp)
        self.lp_map[lp.lp_id] = lp
        # rebuilt with the new LP on the next peek()
        self.heads = None

    def peek(self):
        """Return the LP holding the lowest timestamp event, or None if all are empty.

        self.heads is maintained lazily.  Each LP with events has one live entry,
        keyed by lp.head_key, which is never later than its head:
        - when the head moves earlier (add_event), head_added() pushes a new live
          entry, and the LP's previous one is superseded and dropped when it reaches
          the top;
        - when the head moves later (an event is popped or purged), the live entry
          is left behind and moved to the new head here when it reaches the top.
        The first live entry that matches its LP's head is then the global minimum.
        """
        if self.heads is None:
            self.heads = []
            for i, lp in enumerate(self.lp):
                e = lp.peek()
                lp.head_key = None if e is None else e.timestamp
                if e is not None:
                    self.heads.append((e.timestamp, i))
                lp.tracks_head = True
            heapq.heapify(self.heads)

        heads = self.heads
        while heads:
            ts, i = heads[0]
            lp = self.lp[i]
            if ts != lp.head_key:
                heapq.heappop(heads)
                continue
            e = lp.peek()
            if e is None:
                heapq.heappop(heads)
                lp.head_key = None
            elif e.timestamp == ts:
                return lp
            else:
                lp.head_key = e.timestamp
                heapq.heapreplace(heads, (e.timestamp, i))
        return None

    def head_added(self, lp) -> None:
        """Called by an LP when an event is added at the front of its event list."""
        lp.head_key = lp.event_list[0].timestamp
        heapq.heappush(self.heads, (lp.head_key, lp.pe_index))

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
