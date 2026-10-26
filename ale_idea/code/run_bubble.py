"""The bubble on the dumbbell as a quantics tensor network: the example of the notes, end to end.

    python run_bubble.py                # Y- by reflecting Y+ (one zip-up in step 4)
    python run_bubble.py --subtractor   # also Y- by its own zip-up, as a check of the reflection

G_q(z) = 1/(z - 3 cos q) is given on the dumbbell C_lam (lam = 0.2, N = 512 nodes) for N_q = 1024
momenta; chi_Q is computed on the dumbbell C_mu (mu = 0.4) for all transfer momenta at once and
compared with the Lindhard function (Eq. F3). The script prints the bonds of every step before and
after recompression (Table III) and the error per transfer momentum (Table IV).
"""
import argparse
import time

import numpy as np

import qtt
from dumbbell import chi_contour, fermi, phi_grid, quadrature_weights
from kernel import compressed_kernel, fourier_coefficients, mode_window
from network import (attach_core, attach_output_leaf, chi_from_train, input_leaf, input_train,
                     output_leaf)

# ---------------- parameters of the example (Sec. II A, App. C) ---------------- #

LAM, MU = 0.2, 0.4                     # levels of the two dumbbells
R, RQ, RN = 9, 10, 8                   # bits of the contour points, of the momenta, of the modes n
N, NQ = 2**R, 2**RQ
N_WINDOW = mode_window(LAM, N)         # modes n of G with s_n >= 1e-8 (App. H): [-10, 93]
L_WINDOW = np.arange(-32, 96)          # modes l of chi kept, on the R bits of k
TOL = dict(kernel=1e-8, input=1e-6, leaves=1e-10, dft=1e-12, intermediates=1e-9)


def lindhard(eps_q, w):
    """The Lindhard bubble (Eq. F3) at the complex frequencies w, for every transfer momentum: [Q, k]."""
    f = fermi(eps_q)
    out = np.empty((len(eps_q), len(w)), dtype=complex)
    for Q in range(len(eps_q)):
        eps_qQ, f_qQ = np.roll(eps_q, -Q), np.roll(f, -Q)               # eps_{q+Q}, f(eps_{q+Q})
        out[Q] = np.mean((f - f_qQ)[None, :] / (w[:, None] + (eps_q - eps_qQ)[None, :]), axis=1)
    return out


def report(step, raw_bonds, tt, seconds):
    """One row of Table III: the bonds before and after recompression, and the time of the step."""
    print(f'{step:28s} {seconds:6.1f} s\n    before {raw_bonds}\n    after  {qtt.bonds(tt)}', flush=True)


def main(run_subtractor):
    # ---------------- data and kernel (Secs. II and III) ---------------- #
    t0 = time.perf_counter()
    z, _, _ = quadrature_weights(LAM, N)
    w = chi_contour(MU, N)
    eps_q = 3 * np.cos(2 * np.pi * np.arange(NQ) / NQ)
    G = 1 / (z[:, None] - eps_q[None, :])                              # G_q(z_j), [j, q]
    ghat = fourier_coefficients(G)[np.mod(N_WINDOW, N)]               # [n, q] on the window
    core, U1, U2, U3, s = compressed_kernel(LAM, MU, N, N_WINDOW, L_WINDOW, TOL['kernel'])
    print(f'kernel and HOSVD: ranks {core.shape}, {time.perf_counter() - t0:.1f} s')

    # ---------------- the tensors (App. C) ---------------- #
    t0 = time.perf_counter()
    ghat_tt, site_types = input_train(ghat, N_WINDOW, RQ, RN, TOL['input'])
    leaf1 = input_leaf(U1, s, N_WINDOW, RN, TOL['leaves'])
    leaf2 = input_leaf(U2, s, N_WINDOW, RN, TOL['leaves'])
    leaf3 = output_leaf(U3, L_WINDOW, R, TOL['leaves'])
    dft = qtt.dft_mpo(phi_grid(N), TOL['dft'])
    print(f'tensors: {time.perf_counter() - t0:.1f} s')
    print(f'    ghat {qtt.bonds(ghat_tt)}\n    U1/s {qtt.bonds(leaf1)}\n    U2/s {qtt.bonds(leaf2)}'
          f'\n    U3   {qtt.bonds(leaf3)}\n    DFT  {[W.shape[3] for W in dft[:-1]]}')

    # ---------------- the contraction (App. D, Table III) ---------------- #
    tol = TOL['intermediates']
    start = time.perf_counter()

    t0 = time.perf_counter()
    A = qtt.zip_up_with_leaf(ghat_tt, site_types, leaf1, tol)
    raw = qtt.bonds(A)
    report('1  A(q;a)', raw, qtt.recompress(A, tol), time.perf_counter() - t0)

    t0 = time.perf_counter()
    B = qtt.zip_up_with_leaf(ghat_tt, site_types, leaf2, tol)
    raw = qtt.bonds(B)
    report('2  B(q;b)', raw, qtt.recompress(B, tol), time.perf_counter() - t0)

    t0 = time.perf_counter()
    Bp = attach_core(B, core)
    report("3  B'(q;a,c)", qtt.bonds(Bp), Bp, time.perf_counter() - t0)

    t0 = time.perf_counter()
    Y_plus = qtt.zip_up_correlation(A, Bp, +1, NQ, tol)
    raw = qtt.bonds(Y_plus)
    report('4  Y+(Q;c), zip-up', raw, qtt.recompress(Y_plus, tol), time.perf_counter() - t0)

    t0 = time.perf_counter()
    Y_minus = qtt.apply_mpo(qtt.reflection_mpo(RQ), Y_plus, 1)       # Y-(Q) = Y+(-Q), Eq. (A11)
    raw = qtt.bonds(Y_minus)
    report('   Y-(Q;c), reflection', raw, qtt.recompress(Y_minus, tol), time.perf_counter() - t0)

    if run_subtractor:
        t0 = time.perf_counter()
        Y_check = qtt.zip_up_correlation(A, Bp, -1, NQ, tol)
        raw = qtt.bonds(Y_check)
        report('   Y-(Q;c), zip-up (check)', raw, qtt.recompress(Y_check, tol), time.perf_counter() - t0)
        a, b = qtt.to_dense(Y_minus), qtt.to_dense(Y_check)
        print(f'    reflection vs subtractor zip-up: {np.linalg.norm(a - b) / np.linalg.norm(b):.1e}')

    t0 = time.perf_counter()
    chihat = qtt.add(attach_output_leaf(leaf3, Y_plus, +1), attach_output_leaf(leaf3, Y_minus, -1))
    raw = qtt.bonds(chihat)
    report('5  chihat(Q;l)', raw, qtt.recompress(chihat, tol), time.perf_counter() - t0)

    t0 = time.perf_counter()
    chi = qtt.apply_mpo(dft, chihat, 0)
    raw = qtt.bonds(chi)
    report('6  chi(Q;k)', raw, qtt.recompress(chi, tol), time.perf_counter() - t0)
    print(f'contraction: {time.perf_counter() - start:.1f} s')

    # ---------------- accuracy (App. E, Table IV) ---------------- #
    chi_dense = chi_from_train(chi, R, RQ)                          # [Q, k]
    exact = lindhard(eps_q, w)
    error = np.abs(chi_dense - exact)
    print(f'max error / max |chi| over all Q and k: {error.max() / np.abs(exact).max():.1e}')
    for m in [1, 16, 64, 160, 256, 512]:
        print(f'    Q = 2 pi {m:3d}/1024: error {error[m].max() / np.abs(exact[m]).max():.1e},'
              f' max |chi| {np.abs(exact[m]).max():.2g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subtractor', action='store_true', help='also compute Y- by its own zip-up')
    main(parser.parse_args().subtractor)
