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

    def __init__(self) -> None:
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
