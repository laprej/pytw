from pytw.engine import Event
from pytw.model import Model


class PholdState:
    def __init__(self) -> None:
        self.dummy_state = None


class PholdMessage(Event):
    def __init__(self, ts=None, src=None, dst=None) -> None:
        super().__init__(ts, src, dst)
        self.dummy_data = None


class Phold(Model):
    pass
