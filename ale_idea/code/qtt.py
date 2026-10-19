"""Quantics tensor trains with xfac (Sec. IV and Apps. D to G of the notes).

A train is an xfacpy.TensorTrain_complex. Its cores are numpy arrays of shape
(left bond, physical dimension, right bond); an MPO core has shape (left, out, in, right).
This module builds trains from dense arrays (TT-SVD) and from functions (tensor cross
interpolation), recompresses and adds them, and holds the two zip-up contractions of the network.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('XFACPY_PATH', os.path.expanduser('~/xfac/build/python')))
import xfacpy  # noqa: E402  (compiled module, found through the path above)


# ---------------- building and reading trains ---------------- #

def train(cores):
    """An xfac tensor train with the given cores."""
    tt = xfacpy.TensorTrain_complex(len(cores))
    for i, core in enumerate(cores):
        tt.setCoreAt(i, np.asarray(core, dtype=complex))
    return tt


def cores(tt):
    """The cores of an xfac train, as a list of numpy arrays."""
    return [np.array(core) for core in tt.core]


def bonds(tt):
    """The bond dimensions between neighbouring sites, from left to right."""
    return [core.shape[2] for core in tt.core[:-1]]


def to_dense(tt):
    """Contract a train into the full tensor, one axis per site (for checks on small trains)."""
    full = np.ones((1, 1), dtype=complex)
    for core in cores(tt):
        full = np.tensordot(full, core, axes=([-1], [0]))
    return full.reshape(full.shape[1:-1])


def contract(spec, *tensors):
    """np.einsum with an optimized contraction order, so that the sums run through BLAS."""
    return np.einsum(spec, *tensors, optimize=True)


def rank_for_tolerance(sv, fraction):
    """Smallest rank r such that the discarded singular values carry at most fraction * sum(sv^2)."""
    discarded = np.cumsum((sv**2)[::-1])[::-1]  # discarded[r] = sum over s >= r of sv_s^2
    return max(1, int(np.sum(discarded > fraction * np.sum(sv**2))))


# ---------------- TT-SVD, recompression, sum (Apps. D, F, G) ---------------- #

def recompress(tt, eps):
    """Recompress a train to relative tolerance eps (App. G), with the criterion of Eq. (D1).

    xfac's compressSVD(reltol) keeps at every bond the smallest rank whose discarded weight is at
    most reltol^2 |v|^2. Eq. (D1) allows eps^2/(L-1) per bond, so reltol = eps/sqrt(L-1), and the
    total relative error stays below eps.
    """
    L = len(tt.core)
    if L > 1:
        tt.compressSVD(reltol=eps / np.sqrt(L - 1))
    return tt


def tt_svd(array, eps):
    """TT-SVD (App. D): the train of a dense tensor, one site per axis, at relative tolerance eps.

    The exact train is built by successive QR decompositions and then truncated by xfac's SVD
    sweep with the criterion of Eq. (D1).
    """
    dims = array.shape
    exact, rest, left = [], np.asarray(array, dtype=complex), 1
    for d in dims[:-1]:
        Q, R = np.linalg.qr(rest.reshape(left * d, -1))
        exact.append(Q.reshape(left, d, -1))
        rest, left = R, Q.shape[1]
    exact.append(rest.reshape(left, dims[-1], 1))
    return recompress(train(exact), eps)


def add(tt1, tt2):
    """Sum of two trains with the same sites: the direct sum of the cores (App. F); the bonds add."""
    A, B = cores(tt1), cores(tt2)
    out = []
    for i, (a, b) in enumerate(zip(A, B)):
        if i == 0:
            out.append(np.concatenate([a, b], axis=2))
        elif i == len(A) - 1:
            out.append(np.concatenate([a, b], axis=0))
        else:
            block = np.zeros((a.shape[0] + b.shape[0], a.shape[1], a.shape[2] + b.shape[2]), dtype=complex)
            block[:a.shape[0], :, :a.shape[2]] = a
            block[a.shape[0]:, :, a.shape[2]:] = b
            out.append(block)
    return train(out)


# ---------------- MPOs: adder, reflection, DFT (Sec. IV, App. E) ---------------- #

def adder_tensor(sign=+1):
    """Local tensor of the adder (sign=+1, q' = q + Q) or of the subtractor (sign=-1, q' = q - Q).

    W[c_out, q', q, Q, c_in] = 1 when the bits obey binary addition (subtraction) with the carry
    (borrow) c_in coming from the less significant bit and c_out passed to the more significant
    one (Eq. 37). The carry is the bond of the MPO, so the bond is exactly 2.
    """
    W = np.zeros((2, 2, 2, 2, 2))
    for q in (0, 1):
        for Q in (0, 1):
            for c_in in (0, 1):
                total = q + Q + c_in if sign > 0 else q - Q - c_in
                c_out = total // 2 if sign > 0 else int(total < 0)
                W[c_out, total % 2, q, Q, c_in] = 1
    return W


def reflection_mpo(R):
    """MPO of the reflection Q -> -Q mod 2^R on R bits, most significant bit first.

    Two's-complement negation: complement every bit, then add 1. The "+1" carries a bit from the
    least significant site upward, so the bond is 2. The first core drops the final carry and the
    last core injects c_in = 1. Cores have shape (c_out, out, in, c_in).
    """
    P = np.zeros((2, 2, 2, 2))
    for bit_in in (0, 1):
        x = 1 - bit_in  # bitwise NOT
        for c_in in (0, 1):
            P[x & c_in, x ^ c_in, bit_in, c_in] = 1  # +1: out = x xor c_in, c_out = x and c_in
    mpo = [P.copy() for _ in range(R)]
    mpo[0] = P.sum(axis=0, keepdims=True)  # the carry out of the top bit is discarded
    mpo[-1] = mpo[-1][..., 1:2]            # c_in = 1 at the least significant bit
    return mpo


def apply_mpo(mpo, tt, first_site):
    """Apply an MPO (cores (left, out, in, right)) to consecutive sites of a train.

    This is the plain MPO-MPS product: each core of the result is the core of the MPO times the
    core of the train, so the bonds multiply. The result is not recompressed.
    """
    out = cores(tt)
    for i, W in enumerate(mpo):
        M = out[first_site + i]
        product = contract('aoib,xiy->axoby', W, M)
        a, x, o, b, y = product.shape
        out[first_site + i] = product.reshape(a * x, o, b * y)
    return train(out)


def dft_mpo(phi, eps, sweeps=10, max_bond=64):
    """The inverse DFT F(k, l) = exp(i pi l phi_k / 4) as an MPO, by tensor cross interpolation.

    Site i carries the bit pair (k_{R+1-i}, l_i): l most significant first and k least
    significant first, so that the phases couple nearby sites only (App. E). The local index is
    2*k_bit + l_bit, as for the QFT in the finance project. The interpolation runs a fixed number
    of sweeps (stopping on a small pivot error alone can miss part of an oscillating function),
    is recompressed to eps and checked on random entries.

    Returns:
        list of MPO cores of shape (left, k_bit, l_bit, right).
    """
    N = len(phi)
    R = int(np.log2(N))

    def element(sites):
        k = sum((s // 2) << i for i, s in enumerate(sites))           # k: least significant first
        l = sum((s % 2) << (R - 1 - i) for i, s in enumerate(sites))  # l: most significant first
        l = l - N if l >= N // 2 else l                               # two's complement
        return complex(np.exp(1j * np.pi * l * phi[k] / 4))

    param = xfacpy.TensorCI2Param()
    param.pivot1 = [0] * R
    param.reltol = 1e-14
    param.bondDim = max_bond
    param.fullPiv = True
    tci = xfacpy.TensorCI2_complex(element, [4] * R, param)
    for _ in range(sweeps):
        tci.iterate()
    tt = recompress(tci.tt, eps)
    samples = np.random.default_rng(0).integers(0, 4, size=(2000, R))  # check random entries
    error = max(abs(tt.eval(list(sites)) - element(list(sites))) for sites in samples)
    assert error < 100 * eps, f'DFT MPO not converged: max error {error:.1e}'
    return [core.reshape(core.shape[0], 2, 2, core.shape[2]) for core in cores(tt)]


# ---------------- zip-up contractions (Sec. IV) ---------------- #

def zip_up_with_leaf(ghat, site_types, leaf, eps):
    """Contract the input train with a Tucker leaf over the legs n, by the zip-up algorithm.

    ghat has interleaved sites, site_types lists them ('q' or 'n'); the leaf has the sites
    a, n_R, ..., n_1 in the same order as the n sites of ghat. Padded with identities on the q sites,
    the leaf acts on ghat as an MPO that sums the legs n and puts the Tucker leg a in front.
    The zip-up runs from left to right: it absorbs a q site of ghat and the n site that follows,
    then splits off a finished core by a truncated SVD and carries the rest to the right. The
    untruncated product is never formed, and every core of the result carries one q bit.

    Returns:
        the train with the sites a, q_1, ..., q_Rq, before recompression.
    """
    G, U = cores(ghat), cores(leaf)
    out = [U[0]]  # the leg a
    u0 = U[0].shape[2]
    carry = np.eye(u0, dtype=complex).reshape(u0, 1, u0)  # [output bond, ghat bond, leaf bond]
    fraction = eps**2 / (site_types.count('q') - 1)       # truncation budget per SVD
    next_leaf, i = 1, 0
    while i < len(G):
        X = contract('kgu,gqh->kquh', carry, G[i])  # absorb a q site; q stays open
        i += 1
        if i < len(G) and site_types[i] == 'n':  # absorb the n site of ghat and leaf
            X = contract('kquh,hnm->kqunm', X, G[i])
            X = contract('kqunm,unv->kqmv', X, U[next_leaf])  # sum over the leg n
            i += 1
            next_leaf += 1
        else:
            X = X.transpose(0, 1, 3, 2)  # no n site left: [k, q, ghat, leaf]
        k, q, g, u = X.shape
        if i == len(G):  # the last site closes the train
            out.append(X.reshape(k, q, g * u))
            break
        Usv, sv, Vh = np.linalg.svd(X.reshape(k * q, g * u), full_matrices=False)
        r = rank_for_tolerance(sv, fraction)
        out.append(Usv[:, :r].reshape(k, q, r))
        carry = (sv[:r, None] * Vh[:r]).reshape(r, g, u)
    return train(out)


def zip_up_correlation(A, Bp, sign, n_q, eps):
    """Step 4: the cyclic correlation Y(Q; c) = (1/N_q) sum_q A(q; a) B'(q + sign*Q; a, c).

    A has the sites a, q_1, ..., q_R and B' the sites (a, c), q'_1, ..., q'_R, most significant
    bit first. The adder (sign=+1) or subtractor (sign=-1) joins q_i, q'_i and Q_i at every site;
    the legs a, q_i and q'_i are summed, c and Q_i stay open. The zip-up runs from the most
    significant bit down. The carried tensor holds the output bond, the bond of A, the carry of
    the adder and the bond of B'; after every site a truncated SVD splits off a finished core.

    Returns:
        the train Y with the sites c, Q_1, ..., Q_R, before recompression.
    """
    W = adder_tensor(sign)  # [c_out, q', q, Q, c_in]
    Ac, Bc = cores(A), cores(Bp)
    R = len(Ac) - 1
    r1 = Ac[0].shape[1]
    r3 = Bc[0].shape[1] // r1
    fraction = eps**2 / R  # R SVDs in total

    # first site: sum over the Tucker leg a; the open leg c starts the result
    B0 = Bc[0].reshape(r1, r3, Bc[0].shape[2])       # [a, c, bond of B']
    X = contract('ax,acy->cxy', Ac[0][0], B0) / n_q  # [c, bond of A, bond of B']
    Usv, sv, Vh = np.linalg.svd(X.reshape(r3, -1), full_matrices=False)
    r = rank_for_tolerance(sv, fraction)
    out = [Usv[:, :r].reshape(1, r3, r)]
    carry = (sv[:r, None] * Vh[:r]).reshape(r, Ac[0].shape[2], 1, Bc[0].shape[2])  # [k, a, carry, b]

    for i in range(1, R + 1):
        if i == 1:
            Wi = W.sum(axis=0, keepdims=True)  # the carry out of the top bit is dropped
        else:
            Wi = W
        X = contract('kacb,aqx->kcbqx', carry, Ac[i])  # absorb the site of A (leg q)
        X = contract('kcbqx,cpqQd->kbxpQd', X, Wi)     # the adder: q' = p, Q open, new carry d
        X = contract('kbxpQd,bpy->kQxdy', X, Bc[i])    # absorb the site of B' (leg q' = p)
        k, Q, x, d, y = X.shape
        if i == R:  # least significant bit: c_in = 0
            out.append(X[:, :, :, 0, :].reshape(k, Q, x * y))
            break
        Usv, sv, Vh = np.linalg.svd(X.reshape(k * Q, x * d * y), full_matrices=False)
        r = rank_for_tolerance(sv, fraction)
        out.append(Usv[:, :r].reshape(k, Q, r))
        carry = (sv[:r, None] * Vh[:r]).reshape(r, x, d, y)
    return train(out)
