import pytest

from pytw.models.phold import Phold, PholdState
from pytw.optimistic import Optimistic
from pytw.runner import build as runner_build


def build(engine=None, nlps=16, seed=0, **kwargs):
    return runner_build(Phold, PholdState, nlps, seed, kwargs, engine)


def snapshot(pe):
    """Everything a correct run's result depends on, per LP."""
    return [
        (
            lp.events_processed,
            lp.events_sent_remote,
            lp.rng.getstate(),
            sorted(e.timestamp for e in lp.event_list if not e.cancelled),
        )
        for lp in pe.lp
    ]


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("remote", [0.0, 0.25, 1.0])
def test_matches_sequential(batch, remote):
    """Rollbacks must leave every LP exactly where the sequential engine does."""
    seq_engine, seq_pe = build(percent_remote=remote)
    opt_engine, opt_pe = build(Optimistic(batch=batch), percent_remote=remote)

    assert opt_engine.run(opt_pe, 50.0) == seq_engine.run(seq_pe, 50.0)
    assert snapshot(opt_pe) == snapshot(seq_pe)
    if remote > 0:
        # otherwise the test isn't exercising rollback at all
        assert sum(lp.events_rolled_back for lp in opt_pe.lp) > 0


def test_no_remote_no_rollback():
    """With no remote events an LP can never receive a straggler."""
    engine, pe = build(Optimistic(), percent_remote=0.0)
    engine.run(pe, 50.0)
    assert sum(lp.rollbacks for lp in pe.lp) == 0


def test_gvt_reaches_end():
    engine, pe = build(Optimistic())
    engine.run(pe, 50.0)
    assert engine.gvt(pe) >= 50.0
    assert all(e.timestamp < 50.0 for lp in pe.lp for e in lp.processed)
