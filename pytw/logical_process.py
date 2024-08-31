import heapq

from pytw.event import Event


class LogicalProcess:
    """LogicalProcess (LP)

    NOTE: From Fujimoto pg. 52:
    Local Causality Constrain: A discrete-event simulation, consisting of logical
    processes (LPs) that interact exclusively by exchanging time stamped messages
    obeys the local causality constrain if and only if each LP processes events in
    nondecreasing time stamp order.
    """

    def __init__(self, lpid, pe) -> None:
        self.lp_id = lpid
        self.pe = pe
        self.event_list: list[Event] = []

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

    def send_event(self, e: Event):
        """Send event to e.dst_lp

        Args:
            e (Event): Event to send.
        """
        assert e.dst_lp != None
        assert e.src_lp == None

        e.src_lp = self.lp_id
        # don't call isend directly here
        # instead place the event in a buffer
        self.pe.comm.isend(e, 0)
