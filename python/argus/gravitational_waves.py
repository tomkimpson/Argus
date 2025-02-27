"""Module which holds all functions which are related to the properties of the gravitational wave."""

from numpy import sin, cos
import numpy as np


# @njit(fastmath=True)
def _h_amplitudes(h, ι):
    """Calculate the plus/cross amplitude components of the GW.

    Args:
        h (float): A scalar, the dimensionless GW amplitude
        ι (float): A scalar in radians, the inclination angle of the GW source

    Returns
    -------
        h_plus  (float): The + component of the GW amplitude
        h_cross (float): The x component of the GW amplitude

    """
    return h * (1.0 + cos(ι) ** 2), h * (-2.0 * cos(ι))  # hplus,hcross


# @njit(fastmath=True)
def _principal_axes(θ, φ, ψ):
    """Calculate the two principal axes of the GW propagation.

    Args:
        θ (ndarray): An array of length M, the polar angle of the M GW sources in radians
        φ (ndarray): An array of length M, the azimuthal angle of the GW source in radians
        ψ (ndarray): An array of length M, the polarisation angle of the GW source in radians


    Returns
    -------
        m (ndarray):  A vector of length 3, corresponding to a principal axis of the GW
        n (ndarray):  A vector of length 3, corresponding to a principal axis of the GW

    """
    M = len(θ)  #: How many GW sources are there?

    m = np.zeros(
        (M, 3)
    )  # Watch out! lower case m is different from upper case M. Lets change the notation to avoid any confusion. todo
    m[:, 0] = sin(φ) * cos(ψ) - sin(ψ) * cos(φ) * cos(θ)
    m[:, 1] = -(cos(φ) * cos(ψ) + sin(ψ) * sin(φ) * cos(θ))
    m[:, 2] = sin(ψ) * sin(θ)

    # size M GW sources x 3 component directions
    n = np.zeros_like(m)
    n[:, 0] = -sin(φ) * sin(ψ) - cos(ψ) * cos(φ) * cos(θ)
    n[:, 1] = cos(φ) * sin(ψ) - cos(ψ) * sin(φ) * cos(θ)
    n[:, 2] = cos(ψ) * sin(θ)

    return m, n


# @njit(fastmath=True)
def _polarisation_tensors(m, n):
    """Calculate the two polarisation tensors e_+, e_x. See equation 2a,2d of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.104008.

    Args:
        m (ndarray): A vector of length (M,3), corresponding to a principal axis of the GW
        n (ndarray): A vector of length (M,3), corresponding to a principal axis of the GW

    Returns
    -------
        e_plus  (ndarray): A 3x3(xM) array corresponding to the + polarisation
        e_cross (ndarray): A 3x3(xM) array corresponding to the x polarisation

    """
    e_plus = m[:, None] * m[None, :] - n[:, None] * n[None, :]
    e_cross = m[:, None] * n[None, :] + n[:, None] * m[None, :]

    return e_plus, e_cross


class GW:
    """
    For a population of M black holes, calculate the per-pulsar redshift timeseries a^{(n)}(t).

    Arguments:
        `universe_i`: the realisation of the universe, i.e. the BH-BH population
        `PTA`: The PTA configuration used to observe the GWs from the BH-BH population

    """

    def __init__(self, universe_i, PTA):
        """Initialize the class."""
        # Gw parameters
        self.Ω = universe_i.Ω
        self.δ = universe_i.δ
        self.α = universe_i.α
        self.ψ = universe_i.ψ
        self.h = universe_i.h
        self.ι = universe_i.ι
        self.φ0 = universe_i.φ0

        # PSR related quantities
        self.q = PTA.q
        self.t = PTA.t
        self.d = PTA.d

        # Shapes
        self.M, self.T, self.N = universe_i.M, len(PTA.t), PTA.Npsr

    def compute_a(self):
        """Compute the a(t) timeseries."""
        m, n = _principal_axes(
            np.pi / 2.0 - self.δ, self.α, self.ψ
        )  # Get the principal axes. declination converted to a latitude 0-π. Shape (K,3)
        gw_direction = np.cross(
            m, n
        ).T  # The direction of each source. Shape (3,M). Transpose to enable dot product with q vector
        e_plus, e_cross = _polarisation_tensors(
            m.T, n.T
        )  # The polarization tensors. Shape (3,3,K)
        hp, hx = _h_amplitudes(
            self.h, self.ι
        )  # The plus and cross amplitudes. Can also do h_amplitudes(h*Ω**(2/3),ι) to add a frequency dependence
        dot_product = (
            1.0 + self.q @ gw_direction
        )  # .T                     # Shape (N,M)

        # Amplitudes
        Hij_plus = hp * e_plus
        Hij_cross = hx * e_cross

        Fplus = np.einsum("ijm, in, jn -> mn", Hij_plus, self.q.T, self.q.T)
        Fcross = np.einsum("ijm, in, jn -> mn", Hij_cross, self.q.T, self.q.T)

        # Phases
        earth_term_phase = np.outer(self.Ω, self.t).T + self.φ0  # Shape(T,M)
        phase_correction = self.Ω * dot_product * self.d
        pulsar_term_phase = earth_term_phase.T.reshape(
            self.M, self.T, 1
        ) + phase_correction.T.reshape(
            self.M, 1, self.N
        )  # Shape(M,T,N)

        # Trig terms
        cosine_terms = cos(earth_term_phase).reshape(self.T, self.M, 1) - cos(
            pulsar_term_phase
        ).transpose(1, 0, 2)
        sine_terms = sin(earth_term_phase).reshape(self.T, self.M, 1) - sin(
            pulsar_term_phase
        ).transpose(1, 0, 2)

        # Redshift per pulsar per source over time
        zplus = Fplus * cosine_terms
        zcross = Fcross * sine_terms
        z = (zplus + zcross) / (2 * dot_product.T)  # (T,M,N)

        # Put it all together
        a = np.sum(
            z, axis=1
        )  # the GW on the nth pulsar at time t is the sum over the M GW sources. Shape (T,Npsr)

        return a
