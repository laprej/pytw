from random import random

import mpi4py

from pytw.engine import Event
from pytw.logical_process import LogicalProcess
from pytw.model import Model
from pytw.state import State


class PholdState(State):
    def __init__(self) -> None:
        self.dummy_state = None


class PholdMessage(Event):
    def __init__(self, ts=None, src=None, dst=None) -> None:
        super().__init__(ts, src, dst)
        self.dummy_data = None


class Phold(Model):
    def initialize(self):
        super().initialize()

    def event_handler(self, e: Event, s: State, lp: LogicalProcess):
        super().event_handler(e)

    def reverse_event_handler(self, e: Event):
        super().reverse_event_handler(e)
        comm.abort()


if __name__ == "__main__":
    foo = Phold()
