import bisect
import heapq
import random

from pytw.event import Event
from pytw.processing_element import ProcessingElement


class LogicalProcess:
    """LogicalProcess (LP)

    NOTE: From Fujimoto pg. 52:
    Local Causality Constrain: A discrete-event simulation, consisting of logical
    processes (LPs) that interact exclusively by exchanging time stamped messages
    obeys the local causality constraint if and only if each LP processes events in
    nondecreasing time stamp order.
    """

    def __init__(self, lpid, pe: ProcessingElement, seed=None) -> None:
        self.lp_id = lpid
        self.pe = pe
        self.event_list: list[Event] = []
        self.now = 0.0
        self.model = None
        # per-LP stream so results don't depend on event interleaving across LPs;
        # defaults to the LP id so runs are reproducible
        self.rng = random.Random(lpid if seed is None else seed)
        self.events_processed = 0
        # events sent to an LP other than this one
        self.events_sent_remote = 0
        # events processed by forward() in processing order, available for rollback
        self.processed: list[Event] = []
        # the event whose forward handler is running, so sends can be recorded on it
        self.current = None
        self.rollbacks = 0
        self.events_rolled_back = 0

    def add_event(self, e: Event):
        heapq.heappush(self.event_list, e)

    def receive(self, e: Event):
        """Add a delivered event, first rolling back if it is a straggler."""
        if self.processed and self.processed[-1].timestamp > e.timestamp:
            self.rollback(e.timestamp)
        self.add_event(e)

    def _purge(self):
        # cancelled events are removed lazily, when they reach the front of the queue
        while self.event_list and self.event_list[0].cancelled:
            heapq.heappop(self.event_list)

    def next_event(self):
        self._purge()
        try:
            return heapq.heappop(self.event_list)
        except IndexError:
            return None

    def peek(self):
        self._purge()
        try:
            return self.event_list[0]
        except IndexError:
            return None

    def forward(self, e: Event):
        """Run the model's forward handler, saving RNG state for a later rollback."""
        e.rng_state = self.rng.getstate()
        e.sent = []
        sent_before = self.events_sent_remote
        self.current = e
        self.model.event_handler(e, self.model.state, self)
        self.current = None
        self.events_processed += 1
        e.remote_sent = self.events_sent_remote - sent_before
        e.processed = True
        self.processed.append(e)

    def reverse(self, e: Event):
        """Run the model's reverse handler and restore the RNG to its state before e."""
        assert self.processed and self.processed[-1] is e, "events must be reversed newest first"
        self.processed.pop()
        self.model.reverse_event_handler(e, self.model.state, self)
        self.rng.setstate(e.rng_state)
        self.events_processed -= 1
        self.events_sent_remote -= e.remote_sent
        e.processed = False

    def rollback(self, ts: float, inclusive: bool = False):
        """Undo processed events newer than ts (or at ts, if inclusive), newest first.

        Each undone event goes back in the event list and the events it sent are
        cancelled, which may in turn roll back their destination LPs.
        """
        self.rollbacks += 1
        while self.processed and (
            self.processed[-1].timestamp > ts or inclusive and self.processed[-1].timestamp == ts
        ):
            e = self.processed[-1]
            self.now = e.timestamp
            self.reverse(e)
            self.events_rolled_back += 1
            self.add_event(e)
            sent, e.sent = e.sent, []
            for s in sent:
                self.pe.lp_map[s.dst_lp].cancel(s)
        self.now = self.processed[-1].timestamp if self.processed else 0.0

    def fossil_collect(self, gvt: float):
        """Discard processed events older than gvt; they can never be rolled back.

        self.processed is in nondecreasing timestamp order, so they form a prefix.
        """
        i = bisect.bisect_left(self.processed, gvt, key=lambda e: e.timestamp)
        for e in self.processed[:i]:
            # drop references so the saved RNG state and sent events can be freed
            e.rng_state = None
            e.sent = []
        del self.processed[:i]

    def cancel(self, e: Event):
        """Cancel an event sent to this LP, rolling it back first if it was processed."""
        if e.processed:
            self.rollback(e.timestamp, inclusive=True)
        e.cancelled = True

    def send_event(self, e: Event):
        """Send event to e.dst_lp

        Args:
            e (Event): Event to send.
        """
        assert e.dst_lp != None

        e.src_lp = self.lp_id
        if self.current is not None:
            self.current.sent.append(e)
        if e.dst_lp != self.lp_id:
            self.events_sent_remote += 1
        # don't call isend directly here
        # instead place the event in a buffer
        # self.pe.comm.isend(e, 0)
        self.pe.buffer.put(e)
