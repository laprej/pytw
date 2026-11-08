from abc import ABC, abstractmethod

from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.state import State


class Model(ABC):
    """Model base class (abstract)

    Defines the methods required for all models to implement.

    Models run on the conservative engine must set lookahead: the least delay, from
    the sending LP's current time, of any event it sends to another LP.
    """

    lookahead: float = 0.0

    def __init__(self, s: State, lp: LogicalProcess) -> None:
        self.state = s
        self.lp = lp
        lp.model = self

    @abstractmethod
    def event_handler(self, e: Event, s: State, lp: LogicalProcess):
        """Forward event handler.

        Args:
            e (Event): The event to process.
        """
        pass

    @abstractmethod
    def reverse_event_handler(self, e: Event, s: State, lp: LogicalProcess):
        """Reverse event handler

        Required only for the optimistic engine.

        Args:
            e (Event): The event to reverse.
        """
        pass
