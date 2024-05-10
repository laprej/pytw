from queue import PriorityQueue as PQ

from .event import Event


class LogicalProcess:
    def __init__(self) -> None:
        self.event_list = PQ[Event]()

    def add_event(self, e: Event):
        self.event_list.put(e)

    def next_event(self):
        return self.event_list.get()
