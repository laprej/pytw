import pickle

import pytest

from pytw.event import Event


def test_timestamp_required():
    with pytest.raises(TypeError):
        Event()


def test_ordered_by_timestamp():
    assert Event(1.0) < Event(2.0)
    assert not Event(2.0) < Event(1.0)


def test_pickle_keeps_only_portable_fields():
    """Events sent between ranks carry their own fields, not the sender's rollback bookkeeping."""
    e = Event(1.0, src=2, dst=3)
    e.uid = (0, 7)
    e.rng_state = 42
    e.sent = [Event(2.0)]
    e.transmitted = True

    copy = pickle.loads(pickle.dumps(e))
    assert (copy.timestamp, copy.src_lp, copy.dst_lp, copy.uid) == (1.0, 2, 3, (0, 7))
    assert (copy.rng_state, copy.sent, copy.transmitted) == (None, [], False)
