import random
from sys import float_info

import pytest

from pytw.event import Event
from pytw.logical_process import LogicalProcess


@pytest.fixture
def LP():
    lp = LogicalProcess("a", 0)
    for _ in range(3):
        e = Event(ts=random.random())
        lp.add_event(e)
    return lp


def test_next(LP):
    prev = 0.0

    for _ in range(3):
        e = LP.next_event()
        cur = e.timestamp
        assert cur >= prev
        prev = cur
