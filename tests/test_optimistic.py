import pickle
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from pytw.models.phold import Phold, PholdState
from pytw.optimistic import Optimistic
from pytw.runner import build as runner_build

HAVE_MPIEXEC = shutil.which("mpiexec") is not None
if not HAVE_MPIEXEC:
    # a plain skip is easy to miss, and these are the only tests that exercise rollback
    warnings.warn("mpiexec not found: skipping the MPI tests, so rollback between ranks is untested")


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


@pytest.mark.parametrize("gvt_interval", [1, 4])
@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("remote", [0.0, 0.25, 1.0])
def test_matches_sequential(batch, remote, gvt_interval):
    """On one rank, events are processed in timestamp order, so nothing is rolled back."""
    seq_engine, seq_pe = build(percent_remote=remote)
    opt_engine, opt_pe = build(Optimistic(batch=batch, gvt_interval=gvt_interval), percent_remote=remote)

    assert opt_engine.run(opt_pe, 50.0) == seq_engine.run(seq_pe, 50.0)
    assert snapshot(opt_pe) == snapshot(seq_pe)
    assert sum(lp.rollbacks for lp in opt_pe.lp) == 0


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


@pytest.mark.skipif(not HAVE_MPIEXEC, reason="needs mpiexec")
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("batch, remote, gvt_interval", [(16, 0.25, 16), (1, 1.0, 1), (64, 0.25, 4)])
def test_mpi_matches_sequential(tmp_path, ranks, batch, remote, gvt_interval):
    """Stragglers and anti-messages between ranks still leave every LP where sequential does."""
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
            str(gvt_interval),
        ],
        check=True,
        timeout=120,
        # so -m tests.mpi_snapshot resolves wherever pytest was started
        cwd=Path(__file__).parent.parent,
    )
    with open(out, "rb") as f:
        committed, rolled_back, lps = pickle.load(f)
    # otherwise the test isn't exercising rollback at all
    assert rolled_back > 0

    seq_engine, seq_pe = build(percent_remote=remote)
    assert committed == seq_engine.run(seq_pe, 50.0)
    assert lps == snapshot(seq_pe)


def test_gvt_interval_holds_more_history():
    """Fossil collection only happens at GVT, so a longer interval keeps more history."""
    peaks = []
    for interval in (1, 8):
        engine, pe = build(Optimistic(gvt_interval=interval))
        engine.run(pe, 100.0)
        peaks.append(engine.peak_history)
    assert peaks[1] > peaks[0]


def test_bad_gvt_interval():
    with pytest.raises(ValueError):
        Optimistic(gvt_interval=0)
