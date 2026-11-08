import pytest

from pytw.conservative import Conservative
from pytw.event import Event
from pytw.model import Model
from pytw.models.airport import Airport, AirportState
from pytw.models.phold import Phold, PholdState
from pytw.runner import build as runner_build
from tests.mpi_snapshot import requires_mpiexec, run_mpi, snapshot

MODELS = {"phold": (Phold, PholdState, 50.0), "airport": (Airport, AirportState, 200.0)}


def build(model="phold", engine=None, nlps=16, seed=0, **kwargs):
    model_cls, state_cls, _ = MODELS[model]
    return runner_build(model_cls, state_cls, nlps, seed, kwargs, engine)


@pytest.mark.parametrize(
    "model, kwargs",
    [
        ("phold", {"percent_remote": 0.0}),
        ("phold", {"percent_remote": 0.25}),
        ("phold", {"percent_remote": 1.0}),
        ("phold", {"lookahead": 0.1}),
        ("airport", {}),
        ("airport", {"min_flight_time": 0.25}),
    ],
)
def test_matches_sequential(model, kwargs):
    end = MODELS[model][2]
    seq_engine, seq_pe = build(model, **kwargs)
    con_engine, con_pe = build(model, Conservative(), **kwargs)

    assert con_engine.run(con_pe, end) == seq_engine.run(seq_pe, end)
    assert snapshot(con_pe) == snapshot(seq_pe)


def test_windows():
    """Each window advances by at least the lookahead, so a smaller one needs more of them."""
    windows = []
    for lookahead in (1.0, 0.25):
        engine, pe = build(engine=Conservative(), lookahead=lookahead)
        engine.run(pe, 50.0)
        assert engine.lookahead == lookahead
        assert 1 < engine.windows <= 50.0 / lookahead
        windows.append(engine.windows)
    assert windows[1] > windows[0]


def test_no_lookahead():
    engine, pe = build(engine=Conservative(), lookahead=0.0)
    with pytest.raises(ValueError):
        engine.run(pe, 50.0)


class Liar(Model):
    """Passes a token around a ring of LPs sooner than its declared lookahead."""

    lookahead = 1.0

    def __init__(self, s, lp, nlps) -> None:
        super().__init__(s, lp)
        self.nlps = nlps
        if lp.lp_id == 0:
            lp.send_event(Event(0.0, dst=0))

    def event_handler(self, e, s, lp):
        lp.send_event(Event(lp.now + 0.5, dst=(lp.lp_id + 1) % self.nlps))

    def reverse_event_handler(self, e, s, lp):
        pass


def test_lookahead_violation():
    """A model sending sooner than its declared lookahead is caught, even on one rank."""
    engine, pe = runner_build(Liar, PholdState, 4, 0, {}, Conservative())
    with pytest.raises(RuntimeError, match="lookahead"):
        engine.run(pe, 50.0)
    # the check is only active while the conservative engine runs
    assert pe.lookahead is None


@requires_mpiexec
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize(
    "model, kwargs",
    [("phold", {"percent_remote": 0.25}), ("phold", {"percent_remote": 1.0}), ("airport", {})],
)
def test_mpi_matches_sequential(tmp_path, ranks, model, kwargs):
    end = MODELS[model][2]
    committed, rolled_back, lps = run_mpi(tmp_path, ranks, "conservative", model, 16, end, **kwargs)
    assert rolled_back == 0

    seq_engine, seq_pe = build(model, **kwargs)
    assert committed == seq_engine.run(seq_pe, end)
    assert lps == snapshot(seq_pe)
