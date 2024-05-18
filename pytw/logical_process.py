from queue import PriorityQueue as PQ

from .event import Event


class LogicalProcess:
    """LogicalProcess (LP)

    NOTE: From Fujimoto pg. 52:
    Local Causality Constrain: A discrete-event simulation, consisting of logical
    processes (LPs) that interact exclusively by exchanging time stamped messages
    obeys the local causality constrain if and only if each LP processes events in
    nondecreasing time stamp order.
    """

    def __init__(self) -> None:
        self.event_list = PQ[Event]()

    def add_event(self, e: Event):
        self.event_list.put(e)

    def next_event(self):
        return self.event_list.get()
