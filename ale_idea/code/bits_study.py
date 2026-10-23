"""Do the bond dimensions saturate? The core C and the input ghat against the number of bits of n and l.

With R bits a mode index covers its two's-complement range [-2^(R-1), 2^(R-1)). The kernel is built
once on all N = 512 modes (9 bits for n and for l) and restricted to each pair of windows (R_n, R_l);
the leg dimensions (r1, r2, r3) of the core C are the Tucker ranks of the weighted kernel at 1e-8
(App. A 3). The input ghat is compressed as in the network (interleaved bits, TT-SVD) for R_n up
to 11 bits, from the data on N = 2048 nodes.

    python bits_study.py        # about 10 min; writes bits_study.json and bits_study.{png,pdf}
"""
import json
import time

import matplotlib
import numpy as np

import qtt
from dumbbell import quadrature_weights
from kernel import coefficient_weights, fourier_coefficients, mode_kernel
from network import input_train

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

LAM, MU, RQ = 0.2, 0.4, 10
EPS_KERNEL = 1e-8


def window(bits):
    """The two's-complement range of a mode index with the given number of bits."""
    return np.arange(-2**(bits - 1), 2**(bits - 1))


def tucker_ranks(T, eps):
    """Tucker ranks of T at relative tolerance eps: eps^2/3 per mode, as in kernel.truncated_hosvd."""
    ranks = []
    for mode in range(3):
        sv = np.linalg.svd(np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1), compute_uv=False)
        discarded = np.cumsum((sv**2)[::-1])[::-1]
        ranks.append(max(1, int(np.sum(discarded > eps**2 / 3 * np.sum(sv**2)))))
    return ranks


def core_study(N=512, bits=range(4, 10)):
    """Leg dimensions of the core C for every pair (R_n, R_l), from one kernel on all N modes."""
    all_modes = window(int(np.log2(N)))
    t0 = time.perf_counter()
    K = mode_kernel(LAM, MU, N, all_modes, all_modes)              # [n', n, l]
    s = coefficient_weights(all_modes, LAM)
    K *= s[:, None, None] * s[None, :, None]                       # the weighted kernel, in place
    print(f'kernel on all {N} modes: {time.perf_counter() - t0:.0f} s', flush=True)
    ranks = {}
    for Rn in bits:
        for Rl in bits:
            rows_n, rows_l = window(Rn) + N // 2, window(Rl) + N // 2  # positions in all_modes
            full = len(rows_n) == N and len(rows_l) == N
            sub = K if full else K[np.ix_(rows_n, rows_n, rows_l)]
            ranks[(Rn, Rl)] = tucker_ranks(sub, EPS_KERNEL)
            print(f'R_n = {Rn}, R_l = {Rl}: (r1, r2, r3) = {tuple(ranks[(Rn, Rl)])}', flush=True)
    return ranks


def ghat_study(N=2048, bits=range(4, 12), tolerances=(1e-6, 1e-10)):
    """Largest bond of the input train ghat[q, n] for every number of bits R_n of n."""
    z, _, _ = quadrature_weights(LAM, N)
    eps_q = 3 * np.cos(2 * np.pi * np.arange(2**RQ) / 2**RQ)
    coefficients = fourier_coefficients(1 / (z[:, None] - eps_q[None, :]))     # [n, q], FFT order
    largest = {}
    for Rn in bits:
        modes = window(Rn)
        for eps in tolerances:
            train, _ = input_train(coefficients[np.mod(modes, N)], modes, RQ, Rn, eps)
            largest[(Rn, eps)] = max(qtt.bonds(train))
        print(f'R_n = {Rn}: largest bond of ghat ' +
              ', '.join(f'{largest[(Rn, e)]} at {e:.0e}' for e in tolerances), flush=True)
    return largest


def plot(ranks, largest, path):
    """Left: the legs of C against the bits of their index. Right: the largest bond of ghat."""
    bits = sorted({Rn for Rn, _ in ranks})
    top = max(bits)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    left.plot(bits, [ranks[(R, top)][0] for R in bits], 'o-', label=f"$r_1$ (leg $a$, mode $n'$), $R_\\ell={top}$")
    left.plot(bits, [ranks[(R, top)][1] for R in bits], 's--', label=f'$r_2$ (leg $b$, mode $n$), $R_\\ell={top}$')
    left.plot(bits, [ranks[(top, R)][2] for R in bits], '^-', label=f'$r_3$ (leg $c$, mode $\\ell$), $R_n={top}$')
    left.set_xlabel('bits of the mode index ($R_n$ or $R_\\ell$)')
    left.set_ylabel('leg dimension of $C$')
    left.set_title(f'core $C$, kernel at $10^{{-8}}$', fontsize=10)
    left.legend(fontsize=8)
    left.grid(alpha=0.3)
    gbits = sorted({Rn for Rn, _ in largest})
    for eps, marker in zip(sorted({e for _, e in largest}, reverse=True), ['o-', 's--']):
        right.plot(gbits, [largest[(R, eps)] for R in gbits], marker, label=f'TT-SVD at {eps:.0e}')
    right.set_xlabel('bits of $n$ ($R_n$)')
    right.set_ylabel('largest bond of $\\hat g$')
    right.set_title('input $\\hat g_{q,n}$, 10 bits of $q$', fontsize=10)
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=130)


if __name__ == '__main__':
    largest = ghat_study()
    ranks = core_study()
    json.dump({'core_ranks': {f'{a},{b}': r for (a, b), r in ranks.items()},
               'ghat_largest_bond': {f'{a},{e:.0e}': v for (a, e), v in largest.items()}},
              open('bits_study.json', 'w'), indent=1)
    plot(ranks, largest, 'bits_study')
    print('saved bits_study.json and bits_study.{png,pdf}')
