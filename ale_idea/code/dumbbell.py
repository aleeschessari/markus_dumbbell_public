"""The dumbbell contours, their nodes and the quadrature weights (Secs. III A and III D of the notes).

The dumbbell at level lam is the image of the horizontal line zeta = phi + i*tau*lam under

    omega(zeta) = kappa * sd(K/2 * (zeta + 1) | kappa^2),    kappa = 3/sqrt(10),

a closed curve around the band [-3, 3] that leaves out the poles of the Fermi function at beta = pi.
The contour of chi uses the doubled map 2*omega, around [-6, 6].
"""
import numpy as np
from scipy.special import ellipj, ellipk, ellipkm1

# ---------------- constants of the map (Eq. 2) ---------------- #

KAPPA2 = 0.9                    # parameter m = kappa^2 of the elliptic functions
KAPPA = np.sqrt(KAPPA2)         # 3/sqrt(10)
K = ellipk(KAPPA2)              # K(kappa)  = 2.5781
K_PRIME = ellipkm1(KAPPA2)      # K(kappa') = 1.6124  (scipy: ellipkm1(m) = K(1 - m))
TAU = 2 * K_PRIME / K           # 1.2509: zeta = phi + i*TAU*lam
ALPHA = np.pi * TAU / 4         # 0.9824: decay rate of the Fourier modes per unit of level
BETA = np.pi                    # the inverse temperature for which this map is built


# ---------------- the map ---------------- #

def jacobi_sn_cn_dn(u, m):
    """Jacobi elliptic functions sn, cn, dn at a complex argument u = x + iy, parameter m.

    scipy evaluates them only at real arguments, so we use the addition formulas
    (Abramowitz and Stegun 16.21) with the real functions at (x | m) and at (y | 1 - m).
    """
    u = np.asarray(u, dtype=complex)
    s, c, d, _ = ellipj(u.real, m)
    s1, c1, d1, _ = ellipj(u.imag, 1 - m)
    den = c1**2 + m * s**2 * s1**2
    sn = (s * d1 + 1j * c * d * s1 * c1) / den
    cn = (c * c1 - 1j * s * d * s1 * d1) / den
    dn = (d * c1 * d1 - 1j * m * s * c * s1) / den
    return sn, cn, dn


def omega(zeta):
    """The map omega(zeta) (Eq. 2) and its Jacobian J = d omega / d phi (Eq. 4)."""
    v = (K / 2) * (np.asarray(zeta, dtype=complex) + 1)
    sn, cn, dn = jacobi_sn_cn_dn(v, KAPPA2)
    return KAPPA * sn / dn, KAPPA * (K / 2) * cn / dn**2


def phi_grid(N):
    """The uniform grid phi_j = -1 + 8j/N, j = 0, ..., N-1, on one period of the contour."""
    return -1 + 8 * np.arange(N) / N


def fermi(z):
    """Fermi function f(z) = 1/(exp(beta z) + 1), written with tanh to avoid overflow."""
    return 0.5 * (1 - np.tanh(BETA * z / 2))


# ---------------- the two contours and the weights ---------------- #

def g_contour(lam, N):
    """Nodes z_j = omega_lam(phi_j) of the dumbbell of G and the Jacobian J_lam(phi_j) there."""
    return omega(phi_grid(N) + 1j * TAU * lam)


def chi_contour(mu, N):
    """Points w_k = 2 omega_mu(phi_k) of the dumbbell of chi (the doubled map, Eq. 5)."""
    return 2 * omega(phi_grid(N) + 1j * TAU * mu)[0]


def quadrature_weights(lam, N):
    """Nodes and trapezoidal weights of the dumbbell quadrature (Eq. 18).

    With d_j = -(dphi / 2 pi i) J_lam(phi_j), the sum over j of d_j F(z_j) approximates the
    counterclockwise contour integral of F over C_lam divided by 2 pi i (the parametrization in phi
    runs clockwise, hence the minus sign). The weights c_j = d_j f(z_j) include the Fermi function
    and replace the Matsubara sum.

    Returns:
        z, d, c: arrays of length N.
    """
    z, J = g_contour(lam, N)
    d = -(8 / N) / (2j * np.pi) * J
    return z, d, d * fermi(z)
