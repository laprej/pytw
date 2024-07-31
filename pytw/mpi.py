from mpi4py import MPI

from pytw.engine import Engine

# from pytw.logical_process import LogicalProcess


class MPIBase:
    def __init__(self, e: Engine) -> None:
        self.engine = e
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # self.lps: list[LogicalProcess] = []
