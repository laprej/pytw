import pytest

from pytw.rng import SplitMix64


def test_reference_outputs():
    """First outputs for seed 0 from the reference implementation, splitmix64.c."""
    r = SplitMix64(0)
    assert [r.next64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_setstate_replays():
    r = SplitMix64(7)
    state = r.getstate()
    first = [r.random(), r.randrange(10), r.expovariate(2.0)]
    r.setstate(state)
    assert [r.random(), r.randrange(10), r.expovariate(2.0)] == first


def test_string_seeds_are_reproducible():
    assert SplitMix64("a").next64() == SplitMix64("a").next64()
    assert SplitMix64("a").next64() != SplitMix64("b").next64()


def test_random_range_and_mean():
    r = SplitMix64(1)
    xs = [r.random() for _ in range(100_000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert sum(xs) / len(xs) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 1000])
def test_randrange_uniform(n):
    r = SplitMix64(2)
    draws = 2000 * n if n < 100 else 200_000
    counts = [0] * n
    for _ in range(draws):
        counts[r.randrange(n)] += 1
    expected = draws / n
    assert all(abs(c - expected) < 0.25 * expected for c in counts)


def test_randrange_bounds():
    r = SplitMix64(3)
    assert 0 <= r.randrange(1 << 64) < 1 << 64
    for n in (0, -1, (1 << 64) + 1):
        with pytest.raises(ValueError):
            r.randrange(n)


def test_expovariate_mean():
    r = SplitMix64(4)
    xs = [r.expovariate(4.0) for _ in range(100_000)]
    assert all(x >= 0.0 for x in xs)
    assert sum(xs) / len(xs) == pytest.approx(0.25, rel=0.02)
