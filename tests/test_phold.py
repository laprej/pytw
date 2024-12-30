from pytw.logical_process import LogicalProcess as LP
from pytw.models.phold import Phold, PholdState

from .test_pe import PE


def test_phold_sequential(PE):
    l = LP(0, PE)
    s = PholdState()
    foo = Phold(s, l)
