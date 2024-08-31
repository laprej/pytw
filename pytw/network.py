# see https://github.com/ROSS-org/ROSS/blob/master/core/network-mpi.c
# for inspiration for this implementation
from dataclasses import dataclass

from mpi4py import MPI

from pytw.event import Event


@dataclass
class EventStuff:
    e: Event
    req: MPI.Request
    status: MPI.Status


posted_sends: list[Event] = []


def send_finish():
    pass


def test_q(a, b, c) -> int:
    return 0


def send_begin(a):
    pass


def pytw_net_send(e: Event):
    # do
    # {
    #   changed = test_q(&posted_sends, me, send_finish);
    #   changed |= send_begin(me);
    # } while (changed);
    me = e.src_lp.pe
    while True:
        changed = test_q(posted_sends, me, send_finish)
        changed |= send_begin(me)
        if not changed:
            break
