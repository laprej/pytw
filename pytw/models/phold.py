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
    def __init__(self, s, lp) -> None:
        super().__init__(s, lp)
        self.phold_start_events = 5

        for i in range(self.phold_start_events):
            e = Event(random(), self.lp, self.lp)
            self.lp.send_event(e)

    def event_handler(self, e: Event, s: State, lp: LogicalProcess):
        pass

    def reverse_event_handler(self, e: Event, s: State, lp: LogicalProcess):
        lp.pe.comm.Abort()
