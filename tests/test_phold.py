import copy

import pytest

from pytw.models.phold import Phold, PholdState
from pytw.runner import build as runner_build


def build(nlps=16, seed=0, **kwargs):
    return runner_build(Phold, PholdState, nlps, seed, kwargs)


def test_phold_sequential():
    engine, pe = build()
    processed = engine.run(pe, end_time=50.0)

    assert processed > 0
    assert processed == sum(lp.events_processed for lp in pe.lp)
    assert sum(lp.events_sent_remote for lp in pe.lp) > 0
    # PHOLD conserves the event population: each processed event schedules one more
    assert sum(len(lp.event_list) for lp in pe.lp) == 16 * 5
    assert all(e.timestamp >= 50.0 for lp in pe.lp for e in lp.event_list)


def test_phold_deterministic():
    runs = []
    for _ in range(2):
        engine, pe = build(seed=42)
        runs.append((engine.run(pe, end_time=50.0), [lp.events_processed for lp in pe.lp]))
    assert runs[0] == runs[1]


def test_phold_no_remote():
    engine, pe = build(percent_remote=0.0)
    engine.run(pe, end_time=20.0)
    assert all(lp.events_sent_remote == 0 for lp in pe.lp)


@pytest.mark.parametrize("remote", [0.0, 1.0])
def test_phold_reverse(remote):
    """Rolling back restores model state and RNG so re-execution is identical."""
    engine, pe = build(nlps=4, percent_remote=remote)
    pe.deliver()
    lp = pe.lp[0]
    m = lp.model
    before_state = copy.deepcopy(vars(m.state))
    before_counts = (lp.events_processed, lp.events_sent_remote)
    before_rng = lp.rng.getstate()

    e = lp.next_event()
    lp.now = e.timestamp
    lp.forward(e)
    first = pe.buffer.get()

    lp.reverse(e)
    assert vars(m.state) == before_state
    assert (lp.events_processed, lp.events_sent_remote) == before_counts
    assert lp.rng.getstate() == before_rng

    lp.forward(e)
    second = pe.buffer.get()
    assert (first.timestamp, first.dst_lp) == (second.timestamp, second.dst_lp)
