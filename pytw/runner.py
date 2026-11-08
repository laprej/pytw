"""Run a model and report its event rate.

Usage:
    python -m pytw.runner --model phold --nlps 64 --end 200 --percent-remote 0.5
    mpiexec -n 4 pytw --model phold --engine optimistic --nlps 64 --end 200
    mpiexec -n 4 pytw --model airport --engine conservative --nlps 64 --end 200

Model options are generated from the keyword arguments (with defaults) of the
model's constructor, e.g. Phold(..., start_events=5) becomes --start-events.
Engine options (e.g. --batch for --engine optimistic) are generated the same way.
Pass --model <name> [--engine <name>] --help to see them.
"""

import argparse
import importlib.metadata
import inspect
import time

from mpi4py import MPI

from pytw.conservative import Conservative
from pytw.logical_process import LogicalProcess as LP
from pytw.models.airport import Airport, AirportState
from pytw.models.phold import Phold, PholdState
from pytw.optimistic import Optimistic
from pytw.processing_element import ProcessingElement
from pytw.sequential import Sequential

# name -> (model class, state class)
MODELS = {
    "phold": (Phold, PholdState),
    "airport": (Airport, AirportState),
}

ENGINES = {
    "sequential": Sequential,
    "conservative": Conservative,
    "optimistic": Optimistic,
}

# constructor parameters supplied by the runner rather than the command line
RUNNER_PARAMS = {"self", "s", "lp", "nlps"}


def ctor_params(cls):
    """Yield the (name, default) keyword parameters a model or engine accepts from the CLI."""
    for p in inspect.signature(cls.__init__).parameters.values():
        if p.name not in RUNNER_PARAMS and p.default is not inspect.Parameter.empty:
            yield p.name, p.default


def add_ctor_args(parser, cls):
    group = parser.add_argument_group(f"{cls.__name__} options")
    for name, default in ctor_params(cls):
        group.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=type(default),
            default=default,
            help="(default: %(default)s)",
        )


def build(model_cls, state_cls, nlps, seed, model_kwargs, engine=None):
    engine = engine or Sequential()
    pe = ProcessingElement(engine)
    # each rank builds only its own LPs; seeds depend on the LP, not the rank
    for i in range(nlps):
        if pe.rank_of(i) != pe.rank:
            continue
        lp = LP(i, pe, seed=seed * nlps + i)
        pe.add_lp(lp)
        model_cls(state_cls(), lp, nlps, **model_kwargs)
    return engine, pe


def version():
    try:
        return importlib.metadata.version("pytw")
    except importlib.metadata.PackageNotFoundError:
        # running from a checkout that was never installed
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a pytw model and report its event rate.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version()}")
    parser.add_argument("--model", required=True, choices=MODELS, help="model to run")
    parser.add_argument(
        "--engine", default="sequential", choices=ENGINES, help="simulation engine (default: %(default)s)"
    )
    parser.add_argument("--nlps", type=int, default=16, help="number of LPs (default: %(default)s)")
    parser.add_argument("--end", type=float, default=100.0, help="simulation end time (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed (default: %(default)s)")
    parser.add_argument("--repeat", type=int, default=1, help="number of timed runs (default: %(default)s)")

    # first pass: find the model so its options can be added before the real parse
    # (without -h, which would otherwise print help before the model options exist)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=MODELS)
    pre.add_argument("--engine", default="sequential", choices=ENGINES)
    known, _ = pre.parse_known_args(argv)
    if known.model is not None:
        add_ctor_args(parser, MODELS[known.model][0])
    add_ctor_args(parser, ENGINES[known.engine])
    args = parser.parse_args(argv)
    comm = MPI.COMM_WORLD
    if args.engine == "sequential" and comm.Get_size() > 1:
        parser.error("the sequential engine runs on a single rank")
    # every rank runs the simulation; only rank 0 reports
    log = print if comm.Get_rank() == 0 else lambda *a, **k: None

    model_cls, state_cls = MODELS[args.model]
    model_kwargs = {name: getattr(args, name) for name, _ in ctor_params(model_cls)}
    engine_cls = ENGINES[args.engine]
    engine_kwargs = {name: getattr(args, name) for name, _ in ctor_params(engine_cls)}
    log(
        f"{args.model} on {args.engine}, {comm.Get_size()} rank(s): nlps={args.nlps} end={args.end} seed={args.seed} "
        + " ".join(f"{k}={v}" for k, v in (model_kwargs | engine_kwargs).items())
    )

    rates = []
    for i in range(args.repeat):
        engine, pe = build(model_cls, state_cls, args.nlps, args.seed, model_kwargs, engine_cls(**engine_kwargs))
        comm.Barrier()
        t = time.perf_counter()
        n = engine.run(pe, args.end)
        # the run takes as long as its slowest rank
        dt = comm.allreduce(time.perf_counter() - t, op=MPI.MAX)
        rates.append(n / dt)
        log(f"run {i + 1}: {n} events in {dt:.3f}s = {n / dt:,.0f} ev/s")
        if isinstance(engine, Optimistic):
            rollbacks = comm.allreduce(sum(lp.rollbacks for lp in pe.lp))
            rolled_back = comm.allreduce(sum(lp.events_rolled_back for lp in pe.lp))
            peak = comm.allreduce(engine.peak_history, op=MPI.MAX)
            log(
                f"       {rollbacks} rollbacks, {rolled_back} events rolled back, "
                f"efficiency {n / (n + rolled_back):.1%}, peak history {peak} events/rank"
            )
        elif isinstance(engine, Conservative):
            windows = engine.windows
            log(f"       {windows} windows of lookahead {engine.lookahead}, {n / max(windows, 1):,.1f} events/window")

    if args.repeat > 1:
        log(f"best {max(rates):,.0f} ev/s, mean {sum(rates) / len(rates):,.0f} ev/s")


if __name__ == "__main__":
    main()
