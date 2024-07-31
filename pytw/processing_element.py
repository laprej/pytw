from pytw.mpi import MPIBase


class ProcessingElement(MPIBase):
    """Processing Element (PE)

    PEs can be thought of as a CPU containing multiple Logical Processes (LPs).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lp = []
