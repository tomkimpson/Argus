"""Module used to generate fake, synthetic frequency timeseries for testing the pipeline."""

from numpy import sin, cos
import numpy as np
import pandas as pd
import bilby
import logging
import sdeint

from gravitational_waves import GW

# Set logging level for this module
logging.basicConfig(level=logging.INFO)


class BH_population:
    """
    Calculate the population of black holes which constitute the stochastic GW background.

    This involves randomly drawing 7 GW parameters (Ω,h,φ0,ψ,ι,α,δ) for M sources.
    We use the bilby package to do do the random sampling; note that bilby does not currently let a user seed the sampling process.
    In the future it woukd be better to define our own PowerLaw, Sine, etc. distributions
    See e.g. https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/prior/analytical.py.

    """

    def __init__(
        self,
        Ω_power_law_index=None,
        Ω_min=None,
        Ω_max=None,
        M=None,
        parameters_dictionary=None,
    ):
        """Initialize the class."""
        # Assign arguments to class
        self.M = M
        logging.info(f"Generating a GW signal with M = {M}")

        if (
            parameters_dictionary is not None
        ):  # user specifies the parameters themselves by passing a dictionary

            # Manually extract from the dictionary and make them attributes of the class - easier to handle later
            self.Ω = parameters_dictionary["Ω"]
            self.h = parameters_dictionary["h"]
            self.φ0 = parameters_dictionary["φ0"]
            self.ψ = parameters_dictionary["ψ"]
            self.ι = parameters_dictionary["ι"]
            self.δ = parameters_dictionary["δ"]
            self.α = parameters_dictionary["α"]

        else:  # sample randomly

            # Assign arguments to class
            self.alpha = Ω_power_law_index
            self.Ω_min = Ω_min
            self.Ω_max = Ω_max

            # Define priors for GW parameters and sample
            priors = self._gw_priors()
            samples = priors.sample(M)

            # Manually extract from the dictionary and make them attributes of the class - easier to handle later
            self.Ω = samples["Ω"]
            self.h = samples["h"]
            self.φ0 = samples["φ0"]
            self.ψ = samples["ψ"]
            self.ι = samples["ι"]
            self.δ = samples["δ"]
            self.α = samples["α"]

    def _gw_priors(self):
        """Define the priors on the 7 GW parameters.

        Pass 3 arguments: Ω_power_law_index,Ω_min,Ω_max which define the power law prior on Ω.
        Note that h has a unit delta function prior - all sources have the same unit amplitude.
        """
        priors = bilby.core.prior.PriorDict()
        priors["Ω"] = bilby.core.prior.PowerLaw(
            alpha=self.alpha, minimum=self.Ω_min, maximum=self.Ω_max
        )
        priors["h"] = bilby.core.prior.DeltaFunction(1.0)
        priors["φ0"] = bilby.core.prior.Uniform(0.0, 2 * np.pi)
        priors["ψ"] = bilby.core.prior.Uniform(0.0, np.pi)
        priors["ι"] = bilby.core.prior.Sine(0.0, np.pi)
        priors["δ"] = bilby.core.prior.Cosine(-np.pi / 2, np.pi / 2)
        priors["α"] = bilby.core.prior.Uniform(0.0, 2 * np.pi)

        return priors


class Pulsars:
    """A class which defines the pulsars which make up the PTA.

    It takes the following arguments:

        `pulsar_file`: a path to a CSV which holds the ephemeris parameters of N pulsars

        `γp`: A float which specifies the (inverse) mean reversion timescale of the Ornstein Uhlenbeck process. All pulsars take the same value of γp here, but need not generally.

         `σp`: A float which specifies the magnitude of the pulsar stochastic wobbiling. All pulsars take the same value of σp here, but need not generally.

        `σm`: A float which specifies the magnitude of the measurement noise at the detector.

        `dt_weeks`: A float which specifies how frequently the pulsars are observed. All pulsars are observed at the same times.

        `Tobs_years`: A float which specifies the total observation span in years

    ...and a placeholder **kwargs, which is not currently used.

    """

    def __init__(self, pulsar_file, γp, σp, σm, dt_weeks, Tobs_years):
        """Initialize the class."""
        # Define some universal constants
        pc = 3e16  # parsec in m
        c = 3e8  # speed of light in m/s
        week = (
            7 * 24 * 3600
        )  # a week in seconds. These should be defined elsewhere. todo.

        # Read the CSV and Extract the parameters
        pulsars = pd.read_csv(pulsar_file)

        self.f = pulsars["F0"].to_numpy()  # Hz
        self.fdot = pulsars["F1"].to_numpy()  # s^-2
        self.d = pulsars["DIST"].to_numpy() * 1e3 * pc / c  # this is in units of s^-1
        self.δ = pulsars["DECJD"].to_numpy()  # radians
        self.α = pulsars["RAJD"].to_numpy()  # radians
        self.Npsr = len(self.f)

        # Pulsar positions as unit vectors
        self.q = _unit_vector(np.pi / 2.0 - self.δ, self.α)  # 3 rows, N columns

        # For every pulsar let γ be the same for now.
        self.γp = np.ones_like(self.f) * γp

        # Similarly, let every pulsar have the same value of σp for now. todo
        self.σp = np.ones_like(self.f) * σp

        # assign the measurement noise to the class
        self.σm = σm

        # Some useful reshaping for vectorised calculations later
        self.d = self.d.reshape(self.Npsr, 1)

        # Discrete timesteps
        self.dt = dt_weeks * week
        end_seconds = Tobs_years * 365 * 24 * 3600  # from years to second
        self.t = np.arange(0, end_seconds, self.dt)

        self.ephemeris = self.f + np.outer(self.t, self.fdot)


class fake_observations:
    """A class which defines the pulsars which make up the PTA.

        It takes the following arguments:

        `pulsar_file`: a path to a CSV which holds the ephemeris parameters of N pulsars

        `γp`: A float which specifies the (inverse) mean reversion timescale of the Ornstein Uhlenbeck process. All pulsars take the same value of γp

        `dt_weeks`: A float which specifies how frequently the pulsars are observed. All pulsars are observed at the same times.

        `Tobs_years`: A float which specifies the total observation span in years

    ...and a placeholder **kwargs, which is not currently used.

    """

    def __init__(self, universe_i, PTA):
        """Initialize the class."""
        self.universe_i = universe_i
        self.PTA = PTA

    def generate_intrinstic_state_timeseries(self, seed):

        # Generate pulsar frequency timeseries by solving the Ito equaion dx = Ax dt + BdW
        # All pulsars have independent state evolutions so everything is nice and diagonal.
        Amatrix = np.diag(self.PTA.γp)
        Bmatrix = np.diag(self.PTA.σp)

        # Integrate the state equation
        # e.g. https://pypi.org/project/sdeint/
        def f(x, t):
            return Amatrix.dot(x)

        def g(x, t):
            return Bmatrix

        self.generator = np.random.default_rng(seed)  # Random seeding
        initial_f = np.zeros(
            (self.PTA.Npsr)
        )  # All initial heterodyned frequencies are zero by definition

        # Integration step
        self.state_f = sdeint.itoint(
            f, g, initial_f, self.PTA.t, generator=self.generator
        )

    def generate_measured_frequency_timeseries(self):

        # Construct a stochastic GW background
        SGWB = GW(self.universe_i, self.PTA)
        a = SGWB.compute_a()

        assert a.shape == self.state_f.shape

        self.f_measured_no_noise = (1.0 - a) * self.state_f - a * self.PTA.ephemeris

        # #Create some seeded measurement noise and add it on
        measurement_noise = self.generator.normal(
            0, self.PTA.σm, self.f_measured_no_noise.shape
        )  # Measurement noise. Seeded with the same seed as the process noise
        self.f_measured = self.f_measured_no_noise + measurement_noise

        # assign a to the state
        self.a = a


"""
Given a latitude Θ and a longitude φ, get the xyz unit vector which points in that direction
"""


def _unit_vector(Θ, φ):
    qx = sin(Θ) * cos(φ)
    qy = sin(Θ) * sin(φ)
    qz = cos(Θ)
    return np.array([qx, qy, qz]).T  # This has shape (N,3)


# """
# Given an angle α, return the correlation
# """
# def correlation_function(α):

#     with np.errstate(divide='ignore', invalid='ignore'): #ignore the errors that arise from taking np.log(0). These get replaced with 1s
#         bar = (1.0 - np.cos(α))/2
#         out = np.nan_to_num(1.0 + 3.0*bar * (np.log(bar) - (1.0/6.0)),nan=1.0) #replace nans with 1 for when α=0

#     return out


# GW signal
