"""Helpers for checking a model on the optimistic engine across MPI ranks.

pytest can't run a test across several MPI processes, so run_mpi() launches this
file under mpiexec:
    mpiexec -n 4 python -m tests.mpi_snapshot OUT_PATH MODEL NLPS END_TIME BATCH GVT_INTERVAL [name=value ...]
Every rank runs the model; rank 0 pickles the committed and rolled-back counts and
every LP's final state to OUT_PATH.  name=value pairs are model options.
"""

import ast
import pickle
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from mpi4py import MPI

from pytw.optimistic import Optimistic
from pytw.runner import MODELS, build

HAVE_MPIEXEC = shutil.which("mpiexec") is not None
if not HAVE_MPIEXEC:
    # a plain skip is easy to miss, and these are the only tests that exercise rollback
    warnings.warn("mpiexec not found: skipping the MPI tests, so rollback between ranks is untested")

requires_mpiexec = pytest.mark.skipif(not HAVE_MPIEXEC, reason="needs mpiexec")


def snapshot(pe):
    """Everything a correct run's result depends on, by LP id."""
    return {
        lp.lp_id: (
            lp.events_processed,
            lp.events_sent_remote,
            lp.rng.getstate(),
            dict(vars(lp.model.state)),
            sorted((e.timestamp, type(e).__name__) for e in lp.event_list if not e.cancelled),
        )
        for lp in pe.lp
    }


def run_mpi(tmp_path, ranks, model, nlps, end, batch, gvt_interval, **model_kwargs):
    """Run model on `ranks` ranks; return (committed, rolled_back, snapshot)."""
    out = tmp_path / "snapshot.pkl"
    subprocess.run(
        ["mpiexec", "-n", str(ranks), sys.executable, "-m", "tests.mpi_snapshot", str(out), model]
        + [str(nlps), str(end), str(batch), str(gvt_interval)]
        + [f"{k}={v!r}" for k, v in model_kwargs.items()],
        check=True,
        timeout=120,
        # so -m tests.mpi_snapshot resolves wherever pytest was started
        cwd=Path(__file__).parent.parent,
    )
    with open(out, "rb") as f:
        return pickle.load(f)


def main(argv):
    out, model, nlps, end, batch, interval, *options = argv
    model_kwargs = {k: ast.literal_eval(v) for k, v in (o.split("=", 1) for o in options)}
    model_cls, state_cls = MODELS[model]
    engine = Optimistic(batch=int(batch), gvt_interval=int(interval))
    engine, pe = build(model_cls, state_cls, int(nlps), 0, model_kwargs, engine)
    committed = engine.run(pe, float(end))

    comm = MPI.COMM_WORLD
    rolled_back = comm.allreduce(sum(lp.events_rolled_back for lp in pe.lp))
    parts = comm.gather(snapshot(pe), root=0)
    if comm.Get_rank() == 0:
        with open(out, "wb") as f:
            pickle.dump((committed, rolled_back, {k: v for part in parts for k, v in part.items()}), f)


if __name__ == "__main__":
    main(sys.argv[1:])
