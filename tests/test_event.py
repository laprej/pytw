import pytest

from pytw.event import Event


@pytest.fixture
def event():
    e = Event()
    yield e
