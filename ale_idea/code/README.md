# The bubble on the dumbbell with xfac

These scripts carry out the method of `../bubble_tn_method.pdf` for the example of the notes.
G_q(z) = 1/(z - 3 cos q) is given on the dumbbell C_lam (lam = 0.2, N = 512 nodes) for N_q = 1024
momenta, and the bubble chi_Q is computed on the dumbbell C_mu (mu = 0.4) for all transfer momenta
at once. The tensor trains are `xfacpy.TensorTrain_complex` objects, used as in the finance project:
xfac recompresses them (`compressSVD`) and builds the DFT MPO by tensor cross interpolation
(`TensorCI2_complex`), the way the finance code builds its QFT.

## Running

    python run_bubble.py                 # about 75 s
    python run_bubble.py --subtractor    # also Y- by its own zip-up, about 105 s

The run takes 25 s for the kernel and its HOSVD, 7 s for the tensors and 33 s for the contraction
(63 s with the second zip-up); the rest is the comparison with the Lindhard function.

Requirements: Python 3.13 (the version xfacpy is compiled for), numpy, scipy and the compiled
xfacpy module. The scripts look for it in `~/xfac/build/python`; set `XFACPY_PATH` to use another
build.

## Files

| file | part of the notes | contents |
|---|---|---|
| `dumbbell.py` | Secs. III A, III D | the map omega, the nodes z_j and w_k, the weights c_j and d_j |
| `kernel.py` | Secs. III F, III G, App. C | Fourier coefficients, the mode kernel, the weights s_n, the truncated HOSVD |
| `qtt.py` | Sec. IV, Apps. D to G | TT-SVD, recompression, sum of trains, adder, reflection, DFT MPO, the two zip-ups |
| `network.py` | Secs. V, VI | the input train, the leaves, steps 3 and 5 |
| `run_bubble.py` | Secs. II, VI, VII | the example end to end, Tables I and II |

The zip-up is the algorithm of Stoudenmire and White [New J. Phys. 12, 055026 (2010)] for applying
an MPO to an MPS. Steps 1 and 2 apply a Tucker leaf, padded with identities on the q sites, to the
input train; step 4 zips A and B' together through the adder.

## Output of `python run_bubble.py --subtractor`

```
kernel and HOSVD: ranks (31, 33, 44), 25.3 s
tensors: 6.6 s
    ghat [2, 2, 4, 8, 16, 32, 54, 57, 52, 37, 27, 20, 14, 10, 7, 4, 2]
    U1/s [31, 38, 26, 15, 8, 5, 3, 2]
    U2/s [33, 38, 26, 15, 8, 5, 3, 2]
    U3   [2, 2, 3, 4, 8, 14, 25, 45, 44]
    DFT  [2, 4, 8, 11, 11, 8, 4, 2]
1  A(q;a)                       1.1 s
    before [31, 55, 109, 205, 298, 146, 48, 13, 4, 2]
    after  [31, 46, 85, 85, 56, 29, 15, 8, 4, 2]
2  B(q;b)                       1.1 s
    before [33, 55, 108, 207, 298, 146, 47, 11, 4, 2]
    after  [33, 47, 86, 84, 55, 29, 15, 8, 4, 2]
3  B'(q;a,c)                    0.0 s
    before [33, 47, 86, 84, 55, 29, 15, 8, 4, 2]
    after  [33, 47, 86, 84, 55, 29, 15, 8, 4, 2]
4  Y+(Q;c), zip-up             30.3 s
    before [44, 88, 173, 275, 211, 101, 46, 24, 13, 7]
    after  [27, 34, 26, 18, 12, 9, 7, 6, 4, 2]
   Y-(Q;c), reflection          0.0 s
    before [27, 68, 52, 36, 24, 18, 14, 12, 8, 4]
    after  [27, 34, 26, 18, 12, 9, 7, 6, 4, 2]
   Y-(Q;c), zip-up (check)     30.4 s
    before [44, 88, 173, 275, 211, 101, 46, 24, 13, 7]
    after  [27, 34, 26, 18, 12, 9, 7, 6, 4, 2]
    reflection vs subtractor zip-up: 1.7e-10
5  chihat(Q;l)                  0.0 s
    before [4, 4, 6, 8, 16, 28, 50, 90, 54, 68, 52, 36, 24, 18, 14, 12, 8, 4]
    after  [2, 2, 2, 3, 5, 9, 16, 24, 24, 34, 26, 18, 12, 9, 7, 6, 4, 2]
6  chi(Q;k)                     0.0 s
    before [4, 8, 16, 33, 55, 72, 64, 48, 24, 34, 26, 18, 12, 9, 7, 6, 4, 2]
    after  [2, 4, 7, 8, 11, 16, 20, 24, 24, 34, 26, 18, 12, 9, 7, 6, 4, 2]
contraction: 62.9 s
max error / max |chi| over all Q and k: 1.7e-07
    Q = 2 pi   1/1024: error 2e-04, max |chi| 2.6e-05
    Q = 2 pi  16/1024: error 8e-07, max |chi| 0.0064
    Q = 2 pi  64/1024: error 5e-08, max |chi| 0.054
    Q = 2 pi 160/1024: error 9e-09, max |chi| 0.1
    Q = 2 pi 256/1024: error 5e-09, max |chi| 0.13
    Q = 2 pi 512/1024: error 2e-07, max |chi| 0.22
```

The ranks, the bonds of the tensors, the errors and the times agree with the notes. Two details of
Table I differ because of xfac. Its `compressSVD` truncates in two sweeps, the first from the left,
and leaves the train right-canonical. After recompression a few bonds therefore differ by one from
the table (34 instead of 35 in Y and chi). Before recompression the zip-ups of steps 1 and 2 peak at
298 instead of 368, because a left-to-right zip-up truncates closer to optimally when its input is
right-canonical.

`network.input_leaf` sets the rows of the leaves with s_n < 1e-12 to zero, where the coefficients
of the data are rounding noise. The notes do not mention this yet. Without it the error is the same,
1.7e-7, but the largest bond of A and B grows from 85 to 103.
