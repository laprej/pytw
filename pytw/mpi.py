from mpi4py import MPI


class MPIBase:
    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # self.lps: list[LogicalProcess] = []
