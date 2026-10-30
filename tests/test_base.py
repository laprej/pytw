import pytest

# from pytw.models.base import Base
from pytw.base_model import BaseModel


def test_foo():
    b = BaseModel()
    b.run()
