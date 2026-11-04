"""Airport model

The basic airport model from Fujimoto, "Parallel and Distributed Simulation
Systems" (2000).  Each LP is an airport with a single runway.  An aircraft
arrives, waits for the runway, takes runway_time to land, spends ground_time on
the ground, then departs for a uniformly random other airport.  Flight times are
min_flight_time (the model's lookahead) plus an exponential, so they average
mean_flight_time.

Arrival:   in_the_air += 1; if the runway is free, take it and schedule Landed
           at now + runway_time.
Landed:    in_the_air -= 1, on_the_ground += 1, schedule Departure at
           now + ground_time; then land the next waiting aircraft at
           now + runway_time, or free the runway if none are waiting.
Departure: on_the_ground -= 1, schedule Arrival at another airport.
"""

from pytw.event import Event
from pytw.logical_process import LogicalProcess
from pytw.model import Model
from pytw.state import State


class AirportState(State):
    def __init__(self) -> None:
        # aircraft landing or waiting to land
        self.in_the_air = 0
        self.on_the_ground = 0
        self.runway_free = True
        self.arrivals = 0
        self.landings = 0
        self.departures = 0


class AirportEvent(Event):
    pass


class Arrival(AirportEvent):
    pass


class Landed(AirportEvent):
    pass


class Departure(AirportEvent):
    pass


class Airport(Model):
    def __init__(
        self,
        s: AirportState,
        lp: LogicalProcess,
        nlps: int,
        planes_per_airport: int = 4,
        runway_time: float = 0.5,
        ground_time: float = 2.0,
        min_flight_time: float = 1.0,
        mean_flight_time: float = 6.0,
    ) -> None:
        super().__init__(s, lp)
        if mean_flight_time <= min_flight_time:
            raise ValueError("mean_flight_time must be greater than min_flight_time")
        self.nlps = nlps
        self.runway_time = runway_time
        self.ground_time = ground_time
        self.min_flight_time = min_flight_time
        self.mean_flight_time = mean_flight_time
        self.handlers = {
            Arrival: (self.arrival, self.reverse_arrival),
            Landed: (self.landed, self.reverse_landed),
            Departure: (self.departure, self.reverse_departure),
        }

        # every aircraft starts in flight to its home airport
        for _ in range(planes_per_airport):
            lp.send_event(Arrival(self.flight_time(lp), dst=lp.lp_id))

    def flight_time(self, lp: LogicalProcess) -> float:
        return self.min_flight_time + lp.rng.expovariate(1.0 / (self.mean_flight_time - self.min_flight_time))

    def event_handler(self, e: AirportEvent, s: AirportState, lp: LogicalProcess):
        self.handlers[type(e)][0](e, s, lp)

    def reverse_event_handler(self, e: AirportEvent, s: AirportState, lp: LogicalProcess):
        # the engine cancels the events sent by the forward handler and restores lp.rng
        self.handlers[type(e)][1](e, s, lp)

    def arrival(self, e: Arrival, s: AirportState, lp: LogicalProcess):
        s.arrivals += 1
        s.in_the_air += 1
        e.runway_was_free = s.runway_free
        if s.runway_free:
            s.runway_free = False
            lp.send_event(Landed(lp.now + self.runway_time, dst=lp.lp_id))

    def reverse_arrival(self, e: Arrival, s: AirportState, lp: LogicalProcess):
        s.runway_free = e.runway_was_free
        s.in_the_air -= 1
        s.arrivals -= 1

    def landed(self, e: Landed, s: AirportState, lp: LogicalProcess):
        s.landings += 1
        s.in_the_air -= 1
        s.on_the_ground += 1
        lp.send_event(Departure(lp.now + self.ground_time, dst=lp.lp_id))
        e.runway_was_free = s.runway_free
        if s.in_the_air > 0:
            lp.send_event(Landed(lp.now + self.runway_time, dst=lp.lp_id))
        else:
            s.runway_free = True

    def reverse_landed(self, e: Landed, s: AirportState, lp: LogicalProcess):
        s.runway_free = e.runway_was_free
        s.on_the_ground -= 1
        s.in_the_air += 1
        s.landings -= 1

    def departure(self, e: Departure, s: AirportState, lp: LogicalProcess):
        s.departures += 1
        s.on_the_ground -= 1
        if self.nlps > 1:
            # a uniformly random airport other than this one
            dst = lp.rng.randrange(self.nlps - 1)
            dst += dst >= lp.lp_id
        else:
            dst = lp.lp_id
        lp.send_event(Arrival(lp.now + self.flight_time(lp), dst=dst))

    def reverse_departure(self, e: Departure, s: AirportState, lp: LogicalProcess):
        s.on_the_ground += 1
        s.departures -= 1
