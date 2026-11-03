from random import Random, random

import pytest

from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.mpi import MPIBase
from pytw.processing_element import ProcessingElement
from pytw.sequential import Sequential


@pytest.fixture
def PE():
    engine = Sequential()
    pe = ProcessingElement(engine)

    lp_count = 16
    element_count = 10
    for i in range(lp_count):
        lp = LogicalProcess("abcdefghijklmnopqrstuvwxyz"[i], pe)
        for _ in range(element_count):
            e = Event(ts=random())
            lp.add_event(e)
        pe.add_lp(lp)

    yield pe


@pytest.mark.parametrize("seed", range(5))
def test_peek_finds_global_minimum(seed):
    """PE.peek() matches a scan of every LP while events are added, popped and cancelled."""
    rng = Random(seed)
    pe = ProcessingElement(Sequential())
    for i in range(8):
        pe.add_lp(LogicalProcess(i, pe))
    live = []

    for _ in range(2000):
        op = rng.random()
        if op < 0.5 or not live:
            e = Event(ts=rng.random() * 100, dst=rng.randrange(8))
            pe.lp_map[e.dst_lp].add_event(e)
            live.append(e)
        elif op < 0.6:
            # cancelled events are skipped lazily, when they reach the front of a queue
            e = live.pop(rng.randrange(len(live)))
            e.cancelled = True
        else:
            expected = min(live, key=lambda e: e.timestamp)
            assert pe.peek().peek() is expected
            assert pe.next_event() is expected
            live.remove(expected)

    while live:
        expected = min(live, key=lambda e: e.timestamp)
        assert pe.next_event() is expected
        live.remove(expected)
    assert pe.peek() is None
