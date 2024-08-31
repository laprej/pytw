from abc import ABC, abstractmethod

from mpi4py import MPI

from pytw.event import Event
from pytw.logical_process import LogicalProcess as LP
from pytw.mpi import MPIBase


class Engine(MPIBase, ABC):
    """Engine

    There can only be a single Engine for the entire simulation.  This will determine
    how the simulation behaves: sequential, conversative, or optimistic.
    Engines need to be aware of the MPI details.

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, et) -> None:
        super().__init__()
        self.lp: list[LP] = []
        self.engine_type = et
        if self.engine_type not in ["Sequential", "Conservative", "Optimistic"]:
            raise RuntimeError("bad engine type")
