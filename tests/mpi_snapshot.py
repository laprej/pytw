"""Helpers for checking a model on a parallel engine across MPI ranks.

pytest can't run a test across several MPI processes, so run_mpi() launches this
file under mpiexec:
    mpiexec -n 4 python -m tests.mpi_snapshot OUT_PATH ENGINE MODEL NLPS END_TIME [name=value ...]
Every rank runs the model; rank 0 pickles the committed and rolled-back counts and
every LP's final state to OUT_PATH.  name=value pairs are engine options if the
engine takes them, otherwise model options.
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

from pytw.runner import ENGINES, MODELS, build, ctor_params

HAVE_MPIEXEC = shutil.which("mpiexec") is not None
if not HAVE_MPIEXEC:
    # a plain skip is easy to miss, and these are the only tests that run across ranks
    warnings.warn("mpiexec not found: skipping the MPI tests, so the engines are untested across ranks")

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


def run_mpi(tmp_path, ranks, engine, model, nlps, end, **options):
    """Run model on `ranks` ranks; return (committed, rolled_back, snapshot)."""
    out = tmp_path / "snapshot.pkl"
    subprocess.run(
        ["mpiexec", "-n", str(ranks), sys.executable, "-m", "tests.mpi_snapshot", str(out), engine, model]
        + [str(nlps), str(end)]
        + [f"{k}={v!r}" for k, v in options.items()],
        check=True,
        timeout=120,
        # so -m tests.mpi_snapshot resolves wherever pytest was started
        cwd=Path(__file__).parent.parent,
    )
    with open(out, "rb") as f:
        return pickle.load(f)


def main(argv):
    out, engine, model, nlps, end, *options = argv
    options = {k: ast.literal_eval(v) for k, v in (o.split("=", 1) for o in options)}
    engine_cls = ENGINES[engine]
    engine_names = {name for name, _ in ctor_params(engine_cls)}
    engine_kwargs = {k: v for k, v in options.items() if k in engine_names}
    model_kwargs = {k: v for k, v in options.items() if k not in engine_names}
    model_cls, state_cls = MODELS[model]
    engine, pe = build(model_cls, state_cls, int(nlps), 0, model_kwargs, engine_cls(**engine_kwargs))
    committed = engine.run(pe, float(end))

    comm = MPI.COMM_WORLD
    rolled_back = comm.allreduce(sum(lp.events_rolled_back for lp in pe.lp))
    parts = comm.gather(snapshot(pe), root=0)
    if comm.Get_rank() == 0:
        with open(out, "wb") as f:
            pickle.dump((committed, rolled_back, {k: v for part in parts for k, v in part.items()}), f)


if __name__ == "__main__":
    main(sys.argv[1:])
