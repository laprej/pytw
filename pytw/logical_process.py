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

    def add_event(self, e: Event):
        heapq.heappush(self.event_list, e)

    def next_event(self):
        try:
            return heapq.heappop(self.event_list)
        except IndexError:
            return None

    def peek(self):
        try:
            return self.event_list[0]
        except IndexError:
            return None

    def forward(self, e: Event):
        """Run the model's forward handler, saving RNG state for a later rollback."""
        e.rng_state = self.rng.getstate()
        sent_before = self.events_sent_remote
        self.model.event_handler(e, self.model.state, self)
        self.events_processed += 1
        e.remote_sent = self.events_sent_remote - sent_before

    def reverse(self, e: Event):
        """Run the model's reverse handler and restore the RNG to its state before e."""
        self.model.reverse_event_handler(e, self.model.state, self)
        self.rng.setstate(e.rng_state)
        self.events_processed -= 1
        self.events_sent_remote -= e.remote_sent

    def send_event(self, e: Event):
        """Send event to e.dst_lp

        Args:
            e (Event): Event to send.
        """
        assert e.dst_lp != None

        e.src_lp = self.lp_id
        if e.dst_lp != self.lp_id:
            self.events_sent_remote += 1
        # don't call isend directly here
        # instead place the event in a buffer
        # self.pe.comm.isend(e, 0)
        self.pe.buffer.put(e)
