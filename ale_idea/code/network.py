"""The tensors of the network (App. C of the notes) and the contraction steps (App. D).

All bits are most significant first unless stated otherwise; the modes n and l are stored in
two's complement, i.e. in FFT order.
"""
import numpy as np

from qtt import contract, cores, to_dense, train, tt_svd


# ---------------- the tensors (App. C) ---------------- #

def input_train(ghat, n_window, Rq, Rn, eps):
    """The input ghat[q, n] as a quantics train, from the raw Fourier coefficients (Eq. 29).

    ghat[n, q] holds the coefficients of G_q on the window of modes n. The modes go into 2^Rn
    slots in two's complement, and the bits are interleaved as q_1 n_Rn q_2 n_Rn-1 ...:
    momentum bits most significant first, mode bits least significant first (App. B).

    Returns:
        the train and the list of its site types ('q' or 'n').
    """
    X = np.zeros((2**Rq, 2**Rn), dtype=complex)
    X[:, np.mod(n_window, 2**Rn)] = ghat.T
    T = X.reshape([2] * (Rq + Rn))          # axes q_1 ... q_Rq, n_1 ... n_Rn
    order, types = [], []
    for i in range(max(Rq, Rn)):
        if i < Rq:
            order.append(i)
            types.append('q')
        if i < Rn:
            order.append(Rq + Rn - 1 - i)
            types.append('n')
    return tt_svd(np.transpose(T, order), eps), types


def input_leaf(U, s, n_window, Rn, eps):
    """A leaf U[n, a]/s_n as a train with the sites a, n_Rn, ..., n_1 (the order of the n bits of ghat).

    The weight 1/s_n goes into the leaf, so that the raw ghat can be truncated (App. M, item 1).
    On the window of App. H, s_n >= 1e-8, so 1/s_n does not amplify the rounding errors of U.
    """
    leaf = U / s[:, None]
    full = np.zeros((2**Rn, U.shape[1]), dtype=complex)
    full[np.mod(n_window, 2**Rn)] = leaf
    T = full.reshape([2] * Rn + [U.shape[1]])           # axes n_1 ... n_Rn, a
    return tt_svd(np.transpose(T, [Rn] + list(range(Rn - 1, -1, -1))), eps)


def output_leaf(U3, l_window, R, eps):
    """The output leaf U3[l, c] as a train with the sites l_1, ..., l_R, c (on the bits of k)."""
    full = np.zeros((2**R, U3.shape[1]), dtype=complex)
    full[np.mod(l_window, 2**R)] = U3
    return tt_svd(full.reshape([2] * R + [U3.shape[1]]), eps)


# ---------------- the contraction steps (App. D) ---------------- #

def attach_core(B, core):
    """Step 3: contract the core C[a, b, c] into the first site of B over b.

    The new first site carries the pair (a, c), with a as the slower index.
    """
    Bc = cores(B)
    first = contract('xby,abc->xacy', Bc[0], core)
    x, a, c, y = first.shape
    return train([first.reshape(x, a * c, y)] + Bc[1:])


def attach_output_leaf(L3, Y, sign):
    """Step 5 for one branch: contract the output leaf with Y over c.

    For Y- the sign (-1)^l goes onto the last bit of l, since (-1)^l = (-1)^(l_R).

    Returns:
        the train with the sites l_1, ..., l_R, Q_1, ..., Q_Rq.
    """
    L, Yc = cores(L3), cores(Y)
    leaf_to_y = contract('lc,cy->ly', L[-1][:, :, 0], Yc[0][0])      # sum over c
    last_l = np.tensordot(L[-2], leaf_to_y, axes=([2], [0]))         # the site l_R, joined to Y
    if sign < 0:
        last_l = last_l * np.array([1, -1])[None, :, None]
    return train(L[:-2] + [last_l] + Yc[1:])


def chi_from_train(chi, R, Rq):
    """The dense chi[Q, k] from the result train, whose sites are k_R, ..., k_1, Q_1, ..., Q_Rq."""
    full = to_dense(chi)
    k_axes = list(range(R - 1, -1, -1))                  # put the k bits most significant first
    full = np.transpose(full, k_axes + list(range(R, R + Rq)))
    return full.reshape(2**R, 2**Rq).T
