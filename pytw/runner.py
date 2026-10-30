"""Run a model and report its event rate.

Usage:
    python -m pytw.runner --model phold --nlps 64 --end 200 --percent-remote 0.5

Model options are generated from the keyword arguments (with defaults) of the
model's constructor, e.g. Phold(..., start_events=5) becomes --start-events.
Pass --model <name> --help to see them.
"""
import argparse
import inspect
import time

from pytw.logical_process import LogicalProcess as LP
from pytw.models.phold import Phold, PholdState
from pytw.processing_element import ProcessingElement
from pytw.sequential import Sequential

# name -> (model class, state class)
MODELS = {
    "phold": (Phold, PholdState),
}

# constructor parameters supplied by the runner rather than the command line
RUNNER_PARAMS = {"self", "s", "lp", "nlps"}


def model_params(model_cls):
    """Yield the (name, default) keyword parameters a model accepts from the CLI."""
    for p in inspect.signature(model_cls.__init__).parameters.values():
        if p.name not in RUNNER_PARAMS and p.default is not inspect.Parameter.empty:
            yield p.name, p.default


def add_model_args(parser, model_cls):
    group = parser.add_argument_group(f"{model_cls.__name__} options")
    for name, default in model_params(model_cls):
        group.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=type(default),
            default=default,
            help="(default: %(default)s)",
        )


def build(model_cls, state_cls, nlps, seed, model_kwargs):
    engine = Sequential()
    pe = ProcessingElement(engine)
    for i in range(nlps):
        lp = LP(i, pe, seed=seed * nlps + i)
        pe.add_lp(lp)
        model_cls(state_cls(), lp, nlps, **model_kwargs)
    return engine, pe


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a pytw model and report its event rate.")
    parser.add_argument("--model", required=True, choices=MODELS, help="model to run")
    parser.add_argument("--nlps", type=int, default=16, help="number of LPs (default: %(default)s)")
    parser.add_argument("--end", type=float, default=100.0, help="simulation end time (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed (default: %(default)s)")
    parser.add_argument("--repeat", type=int, default=1, help="number of timed runs (default: %(default)s)")

    # first pass: find the model so its options can be added before the real parse
    # (without -h, which would otherwise print help before the model options exist)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=MODELS)
    known, _ = pre.parse_known_args(argv)
    if known.model is not None:
        add_model_args(parser, MODELS[known.model][0])
    args = parser.parse_args(argv)

    model_cls, state_cls = MODELS[args.model]
    model_kwargs = {name: getattr(args, name) for name, _ in model_params(model_cls)}
    print(
        f"{args.model}: nlps={args.nlps} end={args.end} seed={args.seed} "
        + " ".join(f"{k}={v}" for k, v in model_kwargs.items())
    )

    rates = []
    for i in range(args.repeat):
        engine, pe = build(model_cls, state_cls, args.nlps, args.seed, model_kwargs)
        t = time.perf_counter()
        n = engine.run(pe, args.end)
        dt = time.perf_counter() - t
        rates.append(n / dt)
        print(f"run {i + 1}: {n} events in {dt:.3f}s = {n / dt:,.0f} ev/s")

    if args.repeat > 1:
        print(f"best {max(rates):,.0f} ev/s, mean {sum(rates) / len(rates):,.0f} ev/s")


if __name__ == "__main__":
    main()
