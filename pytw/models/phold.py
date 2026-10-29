"""PHOLD benchmark model

See https://github.com/ROSS-org/ROSS/tree/master/models/phold for the reference
implementation.  Each LP starts with a number of events; every event processed
schedules exactly one new event, either to itself or (with probability
percent_remote) to a uniformly random LP, at now + lookahead + Exp(mean).
"""
from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.model import Model
from pytw.state import State


class PholdState(State):
    pass


class PholdMessage(Event):
    pass


class Phold(Model):
    def __init__(
        self,
        s: PholdState,
        lp: LogicalProcess,
        nlps: int,
        start_events: int = 5,
        mean: float = 1.0,
        lookahead: float = 1.0,
        percent_remote: float = 0.25,
    ) -> None:
        super().__init__(s, lp)
        self.nlps = nlps
        self.phold_start_events = start_events
        self.mean = mean
        self.lookahead = lookahead
        self.percent_remote = percent_remote

        for _ in range(self.phold_start_events):
            e = PholdMessage(lp.rng.expovariate(1.0 / self.mean), dst=lp.lp_id)
            lp.send_event(e)

    def event_handler(self, e: PholdMessage, s: PholdState, lp: LogicalProcess):
        if lp.rng.random() < self.percent_remote:
            dst = lp.rng.randrange(self.nlps)
        else:
            dst = lp.lp_id

        ts = lp.now + self.lookahead + lp.rng.expovariate(1.0 / self.mean)
        lp.send_event(PholdMessage(ts, dst=dst))

    def reverse_event_handler(self, e: PholdMessage, s: PholdState, lp: LogicalProcess):
        # PHOLD has no state to undo.  The engine is responsible for cancelling the
        # event sent by the forward handler, restoring lp.rng, and the LP's counters.
        pass
