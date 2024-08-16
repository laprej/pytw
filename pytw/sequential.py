from pytw.engine import Engine
from pytw.event import Event
from pytw.mpi import MPIBase


class Sequential(Engine):
    def __init__(self) -> None:
        super().__init__("Sequential")

        if self.size > 1:
            # Sequential can only have a single rank
            self.comm.Abort()
