import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pytw.models.phold import Phold, PholdState
from pytw.optimistic import Optimistic
from pytw.runner import build as runner_build


def build(engine=None, nlps=16, seed=0, **kwargs):
    return runner_build(Phold, PholdState, nlps, seed, kwargs, engine)


def snapshot(pe):
    """Everything a correct run's result depends on, by LP id."""
    return {
        lp.lp_id: (
            lp.events_processed,
            lp.events_sent_remote,
            lp.rng.getstate(),
            sorted(e.timestamp for e in lp.event_list if not e.cancelled),
        )
        for lp in pe.lp
    }


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
    # everything processed is older than the final GVT, so it has all been committed
    assert all(not lp.processed for lp in pe.lp)


def test_fossil_collection_bounds_history():
    """The rollback window stays the same size however long the run is."""
    peaks = []
    for end in (100.0, 400.0):
        engine, pe = build(Optimistic())
        committed = engine.run(pe, end)
        peaks.append(engine.peak_history)
    assert committed > 3 * peaks[1]
    # without fossil collection the peak would grow with the run, about 4x here
    assert peaks[1] < 1.5 * peaks[0]


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="needs mpiexec")
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("batch, remote", [(8, 0.25), (1, 1.0)])
def test_mpi_matches_sequential(tmp_path, ranks, batch, remote):
    """Across ranks, with events and anti-messages over MPI, every LP still ends where sequential does."""
    out = tmp_path / "snapshot.pkl"
    subprocess.run(
        [
            "mpiexec",
            "-n",
            str(ranks),
            sys.executable,
            "-m",
            "tests.mpi_snapshot",
            str(out),
            str(batch),
            str(remote),
            "50",
        ],
        check=True,
        timeout=120,
        # so -m tests.mpi_snapshot resolves wherever pytest was started
        cwd=Path(__file__).parent.parent,
    )
    with open(out, "rb") as f:
        committed, lps = pickle.load(f)

    seq_engine, seq_pe = build(percent_remote=remote)
    assert committed == seq_engine.run(seq_pe, 50.0)
    assert lps == snapshot(seq_pe)
