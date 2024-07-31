from random import random

import pytest

from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.mpi import MPIBase
from pytw.processing_element import ProcessingElement as PE
from pytw.sequential import Sequential


@pytest.fixture
def PE():
    engine = MPIBase(PE)
    pe = Sequential(engine)

    lp_count = 16
    element_count = 10
    for i in range(lp_count):
        lp = LogicalProcess("abcdefghijklmnopqrstuvwxyz"[i], pe)
        for _ in range(element_count):
            e = Event(ts=random())
            lp.add_event(e)
        pe.lp.append(lp)

    yield pe
