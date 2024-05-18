from mpi4py import MPI

from pytw.engine import Engine

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Sequential(Engine):
    def __init__(self) -> None:
        super().__init__()

        if size > 1:
            # Sequential can only have a single rank
            comm.Abort()
