from abc import ABC, abstractmethod

from pytw.event import Event


class Model(ABC):
    """Model base class (abstract)

    Defines the methods required for all models to implement.
    """

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def event_handler(self, e: Event):
        """Forward event handler.

        Args:
            e (Event): The event to process.
        """
        pass

    @abstractmethod
    def reverse_event_handler(self, e: Event):
        """Reverse event handler

        Required only for Optimistic engine.

        Args:
            e (Event): The event to reverse.
        """
        pass
