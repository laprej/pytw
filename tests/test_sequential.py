from random import random

import pytest

from pytw.engine import Engine
from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.mpi import MPIBase
from pytw.sequential import Sequential

from .test_pe import PE


def test_seq(PE):
    prev = PE.next_event()

    while last := PE.next_event():
        assert prev.timestamp <= last.timestamp
        prev = last


def test_send(PE):
    """Send an event from one LP to another."""
    e = Event(random(), dst="b")
    PE.lp[0].send_event(e)
