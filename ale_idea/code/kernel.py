"""Fourier modes on the dumbbells, the mode kernel and its compression (Secs. III F, III G and App. C).

The kernel Khat[n', n, l] maps the Fourier coefficients of G on C_lam (n', n) to those of chi on
C_mu (l). It depends on the two contours and on beta only, never on G. It is weighted by the
worst-case size s_n of the coefficients and compressed by a truncated HOSVD into Tucker form,
a core C[a, b, c] and three factors U1[n', a], U2[n, b], U3[l, c].
"""
import numpy as np

from dumbbell import ALPHA, phi_grid, chi_contour, quadrature_weights


def fourier_coefficients(values):
    """Fourier coefficients on a dumbbell along axis 0 (Eq. 23), in FFT order n = 0, 1, ..., -1.

    values[j] = sum_n ghat_n exp(i pi n phi_j / 4) with phi_j = -1 + 8j/N, hence
    ghat_n = exp(i pi n / 4) / N * FFT(values)_n.
    """
    N = values.shape[0]
    n = np.fft.fftfreq(N, 1 / N)
    phase = np.exp(1j * np.pi * n / 4) / N
    return np.fft.fft(values, axis=0) * phase.reshape((N,) + (1,) * (values.ndim - 1))


def mode_kernel(lam, mu, N, n_window, l_window, k_chunk=32):
    """The kernel Khat[n', n, l] of Eq. (A8) for n', n in n_window and l in l_window.

    For every output point w_k the double sum over the nodes (j, m) is a two-dimensional FFT of
    c_j d_m / (z_j - z_m + w_k); the sum over k then projects on the output modes l.
    Cost O(N^3 log N); about 10 s at N = 512.
    """
    z, d, c = quadrature_weights(lam, N)
    w = chi_contour(mu, N)
    rows = np.mod(n_window, N)                       # FFT positions of the modes in the window
    # exp(i pi (n' phi_j + n phi_m) / 4) = exp(-i pi (n' + n) / 4) * exp(2 pi i (n' j + n m) / N)
    phase = np.exp(-1j * np.pi * (n_window[:, None] + n_window[None, :]) / 4)
    project = np.exp(-1j * np.pi * np.outer(phi_grid(N), l_window) / 4) / N      # [k, l]
    kernel = np.zeros((len(n_window), len(n_window), len(l_window)), dtype=complex)
    for k0 in range(0, N, k_chunk):
        ks = np.arange(k0, min(N, k0 + k_chunk))
        block = np.empty((len(n_window), len(n_window), len(ks)), dtype=complex)
        for i, k in enumerate(ks):
            fft2 = np.fft.ifft2(np.outer(c, d) / (z[:, None] - z[None, :] + w[k])) * N * N
            block[:, :, i] = phase * fft2[np.ix_(rows, rows)]
        kernel += np.tensordot(block, project[ks], axes=([2], [0]))
    return kernel


def coefficient_weights(n, lam):
    """The weights s_n of Eq. (C3): the worst-case size of the Fourier coefficient n of G on C_lam.

    Positive modes decay at the rate ALPHA*lam, negative ones at ALPHA*(2 - lam).
    """
    return np.where(n >= 0, np.exp(-ALPHA * lam * n), np.exp(-ALPHA * (2 - lam) * np.abs(n)))


def mode_window(lam, N, s_cut=1e-8):
    """The input window of Sec. III F: the modes n in [-N/2, N/2) with s_n >= s_cut.

    Below s_cut the modes cannot change chi at the tolerance of the kernel, and the HOSVD returns
    their rows of U as rounding noise, which 1/s_n would amplify. For lam = 0.2 this is [-10, 93].
    """
    n = np.arange(-N // 2, N // 2)
    return n[coefficient_weights(n, lam) >= s_cut]


def truncated_hosvd(T, eps):
    """Truncated higher-order SVD of a three-index tensor (Eq. A9).

    Each factor keeps the leading left singular vectors of the mode unfolding, as few as allowed by
    a discarded weight of at most eps^2/3 of |T|^2, so the total relative error is at most eps.

    Returns:
        core, [U1, U2, U3] with T ~ sum_abc core[a, b, c] U1[:, a] U2[:, b] U3[:, c].
    """
    factors = []
    for mode in range(3):
        unfolding = np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)
        U, sv, _ = np.linalg.svd(unfolding, full_matrices=False)
        discarded = np.cumsum((sv**2)[::-1])[::-1]       # discarded[r] = sum over s >= r of sv_s^2
        rank = max(1, int(np.sum(discarded > eps**2 / 3 * np.sum(sv**2))))
        factors.append(U[:, :rank])
    core = T
    for U in factors:                       # each step contracts the leading index with U^dagger
        core = np.tensordot(core, U.conj(), axes=([0], [0]))
    return core, factors


def compressed_kernel(lam, mu, N, n_window, l_window, eps):
    """Build the weighted kernel s_n' s_n Khat (App. C) and compress it by the truncated HOSVD.

    Returns:
        core [a, b, c], U1 [n', a], U2 [n, b], U3 [l, c] and the weights s_n on n_window.
    """
    s = coefficient_weights(n_window, lam)
    weighted = mode_kernel(lam, mu, N, n_window, l_window) * s[:, None, None] * s[None, :, None]
    core, (U1, U2, U3) = truncated_hosvd(weighted, eps)
    return core, U1, U2, U3, s
