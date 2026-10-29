from pytw.engine import Engine
from pytw.event import Event
from pytw.mpi import MPIBase


class Sequential(Engine):
    def __init__(self) -> None:
        super().__init__("Sequential")

        if self.size > 1:
            # Sequential can only have a single rank
            self.comm.Abort()

    def run(self, pe, end_time: float) -> int:
        """Process events in global timestamp order until end_time.

        Args:
            pe (ProcessingElement): The (only) PE, with its LPs and models attached.
            end_time (float): Events with timestamp >= end_time are not processed.

        Returns:
            int: The number of events processed.
        """
        processed = 0
        # deliver any events sent during model initialization
        pe.deliver()

        while (lp := pe.peek()) is not None and lp.peek().timestamp < end_time:
            e = lp.next_event()
            assert e.timestamp >= lp.now, "local causality constraint violated"
            lp.now = e.timestamp
            # call the handler directly: sequential never rolls back, so skip lp.forward()'s state saving
            lp.model.event_handler(e, lp.model.state, lp)
            lp.events_processed += 1
            processed += 1
            pe.deliver()

        return processed
