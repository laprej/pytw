from random import random

import pytest

from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.sequential import Sequential


@pytest.fixture
def PE():
    pe = Sequential()

    lp_count = 16
    element_count = 10
    for _ in range(lp_count):
        lp = LogicalProcess()
        for _ in range(element_count):
            e = Event(ts=random())
            lp.add_event(e)
        pe.lps.append(lp)

    yield pe


def test_seq(PE):
    prev = PE.next_event()

    while last := PE.next_event():
        assert prev.timestamp <= last.timestamp
        prev = last
