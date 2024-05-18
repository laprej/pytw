from abc import ABC

from pytw.event import Event


class Engine(ABC):
    """Engine

    There can only be a single Engine for the entire simulation.  This will determine
    how the simulation behaves: sequential, conversative, or optimistic.

    Args:
        ABC (_type_): _description_
    """

    def __init__(self) -> None:
        self.engine_type = None
        super().__init__()

    def next_event(self) -> Event:
        pass

    def start(self) -> None:
        pass
