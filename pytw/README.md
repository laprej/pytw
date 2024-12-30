Communication Patterns

  _______MPI_______
 /                 \
^                   |
|                   v
PE                  PE
^                   |
|                   v
LP                  LP

There should be one, and only one Engine
PE -> MPIBase, maybe keep an Engine reference
