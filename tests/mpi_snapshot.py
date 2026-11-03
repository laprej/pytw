"""Run PHOLD on the optimistic engine and pickle the committed and rolled-back counts
and every LP's final state to a file.

Launched under mpiexec by test_optimistic.py:
    mpiexec -n 4 python -m tests.mpi_snapshot OUT_PATH BATCH PERCENT_REMOTE END_TIME GVT_INTERVAL
"""

import pickle
import sys

from mpi4py import MPI

from pytw.optimistic import Optimistic
from tests.test_optimistic import build, snapshot

out, batch, remote, end, interval = (
    sys.argv[1],
    int(sys.argv[2]),
    float(sys.argv[3]),
    float(sys.argv[4]),
    int(sys.argv[5]),
)
engine, pe = build(Optimistic(batch=batch, gvt_interval=interval), percent_remote=remote)
committed = engine.run(pe, end)
rolled_back = MPI.COMM_WORLD.allreduce(sum(lp.events_rolled_back for lp in pe.lp))

parts = MPI.COMM_WORLD.gather(snapshot(pe), root=0)
if MPI.COMM_WORLD.Get_rank() == 0:
    with open(out, "wb") as f:
        pickle.dump((committed, rolled_back, {k: v for part in parts for k, v in part.items()}), f)
