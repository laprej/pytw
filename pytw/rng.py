"""SplitMix64 random number generator

Each LP owns one.  Its entire state is a single 64-bit integer, so saving it
before every event (for rollback) is a single int copy, unlike random.Random,
whose Mersenne Twister state is a tuple of 625 ints.  Not for cryptographic use.

See Steele, Lea & Flood, "Fast Splittable Pseudorandom Number Generators"
(OOPSLA 2014), and https://prng.di.unimi.it/splitmix64.c for the reference.
"""

import hashlib
import math

MASK = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    """The subset of the random.Random interface that models use."""

    def __init__(self, seed=0) -> None:
        if not isinstance(seed, int):
            # e.g. string LP ids; hash() is salted per process, so it isn't reproducible
            seed = int.from_bytes(hashlib.sha256(repr(seed).encode()).digest()[:8], "little")
        self.state = seed & MASK

    def next64(self) -> int:
        """Return the next 64-bit output."""
        self.state = z = (self.state + GOLDEN_GAMMA) & MASK
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Return a float uniformly distributed in [0, 1)."""
        return (self.next64() >> 11) * (1.0 / (1 << 53))

    def randrange(self, n: int) -> int:
        """Return an int uniformly distributed in [0, n)."""
        if not 0 < n <= 1 << 64:
            raise ValueError(f"randrange({n}) needs 0 < n <= 2**64")
        # Lemire's multiply-shift, rejecting the few low products that would bias it
        m = self.next64() * n
        if m & MASK < n:
            threshold = ((1 << 64) - n) % n
            while m & MASK < threshold:
                m = self.next64() * n
        return m >> 64

    def expovariate(self, lambd: float) -> float:
        """Return an exponentially distributed float with rate lambd (mean 1 / lambd)."""
        # 1 - random() is in (0, 1], so the log is always defined
        return -math.log(1.0 - self.random()) / lambd

    def getstate(self) -> int:
        return self.state

    def setstate(self, state: int) -> None:
        self.state = state
