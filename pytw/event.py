class Event:
    def __init__(self, ts=None, src=None, dst=None) -> None:
        self.timestamp = None
        self.src_lp = None
        self.dst_lp = None

        if ts:
            self.timestamp = ts
        if src:
            self.src_lp = src
        if dst:
            self.dst_lp = dst

    def __lt__(self, other):
        return self.timestamp < other.timestamp
