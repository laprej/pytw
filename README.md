# pytw

This is a [Time Warp](https://dl.acm.org/doi/abs/10.1145/3916.3988) engine implemented in python.

* LPs drive the simulation
* LPs execute in-order events
* LPs can send messages to other LPs
* LPs can send messages to themselves
* IFF an LP sends a message into another LP's "past" a rollback occurs

## Engines

* `sequential`: one rank, events in global timestamp order
* `conservative`: any number of ranks, synchronous windows bounded by the models' lookahead; never rolls back
* `optimistic`: any number of ranks, Time Warp with rollback, anti-messages and GVT-based fossil collection
