from typing import List

from pytw.logical_process import LogicalProcess


class ProcessingElement:
    """Processing Element (PE)

    PEs can be thought of as a CPU containing multiple Logical Processes (LPs).
    """

    def __init__(self) -> None:
        self.lp = List[LogicalProcess]
