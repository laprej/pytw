import copy

import pytest

from pytw.models.airport import Airport, AirportState, Arrival, Landed
from pytw.optimistic import Optimistic
from pytw.runner import build as runner_build
from tests.mpi_snapshot import requires_mpiexec, run_mpi, snapshot


def build(engine=None, nlps=16, seed=0, **kwargs):
    return runner_build(Airport, AirportState, nlps, seed, kwargs, engine)


def pending(lp, cls):
    return sum(1 for e in lp.event_list if isinstance(e, cls) and not e.cancelled)


@pytest.mark.parametrize("nlps", [1, 16])
def test_aircraft_conserved(nlps):
    engine, pe = build(nlps=nlps, planes_per_airport=4)
    engine.run(pe, 200.0)

    airborne_or_parked = sum(lp.model.state.in_the_air + lp.model.state.on_the_ground for lp in pe.lp)
    in_flight = sum(pending(lp, Arrival) for lp in pe.lp)
    assert airborne_or_parked + in_flight == 4 * nlps
    for lp in pe.lp:
        s = lp.model.state
        # the runway is busy exactly while an aircraft is landing
        assert pending(lp, Landed) == (0 if s.runway_free else 1)
        assert s.arrivals >= s.landings >= s.departures > 0


def drain(pe):
    sent = []
    while not pe.buffer.empty():
        sent.append(pe.buffer.get())
    return sent


def test_reverse_restores_every_event():
    """Forward, reverse and forward again each event: state and RNG come back exactly, and
    re-execution sends the same events.  Covers every event type and runway branch."""
    engine, pe = build(nlps=4)
    pe.deliver()
    branches = set()

    for _ in range(3000):
        lp = pe.peek()
        e = lp.next_event()
        lp.now = e.timestamp
        before = (copy.deepcopy(vars(lp.model.state)), lp.rng.getstate(), lp.events_processed, lp.events_sent_remote)

        lp.forward(e)
        first = [(type(s), s.timestamp, s.dst_lp) for s in drain(pe)]
        lp.reverse(e)
        assert (vars(lp.model.state), lp.rng.getstate(), lp.events_processed, lp.events_sent_remote) == before

        lp.forward(e)
        sent = drain(pe)
        assert [(type(s), s.timestamp, s.dst_lp) for s in sent] == first
        for s in sent:
            pe.lp_map[s.dst_lp].add_event(s)

        if isinstance(e, Arrival):
            branches.add(("arrival", "runway free" if e.runway_was_free else "runway busy"))
        elif isinstance(e, Landed):
            branches.add(("landed", "next lands" if any(t is Landed for t, _, _ in first) else "runway freed"))
        else:
            branches.add(("departure", ""))

    assert len(branches) == 5


@pytest.mark.parametrize("batch, gvt_interval", [(1, 1), (16, 16)])
def test_optimistic_matches_sequential(batch, gvt_interval):
    seq_engine, seq_pe = build()
    opt_engine, opt_pe = build(Optimistic(batch=batch, gvt_interval=gvt_interval))
    assert opt_engine.run(opt_pe, 200.0) == seq_engine.run(seq_pe, 200.0)
    assert snapshot(opt_pe) == snapshot(seq_pe)


@requires_mpiexec
@pytest.mark.parametrize("ranks", [2, 4])
def test_mpi_matches_sequential(tmp_path, ranks):
    """Rollbacks undo real model state here, unlike PHOLD, and must still match sequential."""
    committed, rolled_back, lps = run_mpi(
        tmp_path, ranks, "optimistic", "airport", 16, 200.0, batch=16, gvt_interval=16
    )
    # otherwise the test isn't exercising rollback at all
    assert rolled_back > 0

    seq_engine, seq_pe = build()
    assert committed == seq_engine.run(seq_pe, 200.0)
    assert lps == snapshot(seq_pe)


def test_bad_flight_times():
    with pytest.raises(ValueError):
        build(min_flight_time=5.0, mean_flight_time=5.0)
