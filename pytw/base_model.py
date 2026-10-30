import random

from pytw.engine import Engine
from pytw.logical_process import LogicalProcess


class BaseModel:
    endtime = 10

    def __init__(self):
        self.timestamp = 0.0

    def event(self):
        self.timestamp += random.random()

    def run(self):
        # global endtime
        while self.timestamp < type(self).endtime:
            self.event()
