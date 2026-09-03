# quantized-solver-simulation

Simulation of an int8 iterative-refinement solver. The factorization
is LU with the trailing updates done in max-scaled base-254 slices, a
dial picks the slice depth from the conditioning, and refinement runs
the answer down to the floor (FP64).

Run with

```
python -m pytest -s
```

Tests print their measured numbers before asserting. Several asserts 
guard numbers from the experiment record, and the printed values are 
how you notice a number moved before an assert trips. 

Two rounding modes exist, round-to-nearest by default and stochastic
rounding in sr_slicing. Depth is written k and the per-slice unit is
u_Q = 1/beta with beta = 254.
