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
    for _ in range(lp_count):
        lp = LogicalProcess()
        for _ in range(element_count):
            e = Event(ts=random())
            lp.add_event(e)
        pe.lps.append(lp)

    yield pe
