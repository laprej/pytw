# see https://github.com/ROSS-org/ROSS/blob/master/core/network-mpi.c
# for inspiration for this implementation

from pytw.event import Event


def test_q(a, b, c):
    pass


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
