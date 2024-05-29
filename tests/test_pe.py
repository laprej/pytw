import pytest

from pytw.processing_element import ProcessingElement as PE


@pytest.fixture
def PE():
    pe = PE()
    return pe
