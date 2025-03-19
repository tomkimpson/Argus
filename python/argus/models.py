"""Module for specifying models to be used with a Kalman filter."""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag
from typing import Any, Dict, List
from line_profiler import profile
from argus.jmath import get_F_spin, get_F, get_Q

class ModelHyperClass(ABC):
    """Abstract base class for models used with the Kalman filter.

    Any subclass must implement the following methods:
      - F_matrix(dt: float) -> np.ndarray
      - Q_matrix(dt: float) -> np.ndarray
      - H_matrix() -> List[np.ndarray] (or an object array)
      - R_matrix() -> Any

    Note:
    ----
    In some models the state transition and process-noise matrices may depend
    on the time step dt.

    """

    @abstractmethod
    def F_matrix(self, dt: float) -> np.ndarray:
        """Return the state–transition matrix for time step dt."""
        pass

    @abstractmethod
    def Q_matrix(self, dt: float) -> np.ndarray:
        """Return the process–noise covariance matrix for time step dt."""
        pass

    @abstractmethod
    def H_matrix(self) -> List[np.ndarray]:
        """Return the observation (measurement) matrices."""
        pass

    @abstractmethod
    def R_matrix(self) -> Any:
        """Return the observation noise covariance matrix."""
        pass


class StochasticGWBackgroundModel(ModelHyperClass):
    """A model class for the Stochastic Gravitational Wave Background.

    The state vector for all pulsars is taken to be

        X = [X_GW, X_spin, X_timing]
    with
        X_GW = [r^(1), a^(1),r^(2),a^(2),...,r^(N),a^(N)]

        X_spin = [δφ^(1), δf^(1), δφ^(2), δf^(2),..., δφ^(N), δf^(N)]

        X_timing = [X_timing^(1), X_timing^(2),..., X_timing^(N)]

    and
        X_timing^(n) = [δε_1^(n), δε_2^(n),..., δε_M^(n)]

    where M[n] is the number of extra (design) parameters for that pulsar.

    The measurement equation is

        δt^(n) = (1/f₀)·δφ − r + (design row)·[δε].

    """

    def __init__(
        self,
        df_psr: Any,
        hd_correlation_matrix: np.ndarray,
        pulsar_design_matrices: np.ndarray,
    ) -> None:
        """Initialize the StochasticGWBackgroundModel.

        Parameters
        ----------
        df_psr : DataFrame
            A pandas DataFrame containing pulsar information. The DataFrame is
            assumed to contain at least the following columns:
                - dim_M: integer, the number of design parameters for that pulsar.
                - gamma_p: the spin–noise damping rate.
                - sigma_p: the spin–noise white noise amplitude.
                - f0: the pulsar spin frequency.
                - sigma_t: the measurement noise standard deviation.
        hd_correlation_matrix : np.ndarray
            Precomputed Hellings-Downs correlation matrix
        """
        self.Npsr = len(df_psr)
        print("The number of pulsars is:", self.Npsr)
        self.name = "Stochastic GW background model"
        # Total state dimension: for each pulsar, two state variables from spin noise,
        # two from GW noise, and dim_M extra parameters.
        self.nx = self.Npsr * (2 + 2) + df_psr["dim_M"].sum()
        self.M = df_psr["dim_M"].values.astype(int)  # array of integers
        self.M_sum = self.M.sum()

        self.hd_correlation_matrix = hd_correlation_matrix

        self.M_start_indices = np.cumsum([0] + [m for m in self.M]) + 4 * self.Npsr


        self.M_cumsum = np.concatenate(([0], np.cumsum(self.M)))




        # Used in the H_matrix function
        self.pulsar_design_matrices = pulsar_design_matrices
        self.design_matrix_counter = np.zeros(self.Npsr)
        self.F = None

    def set_global_parameters(self, params: Dict[str, Any]) -> None:
        """Set global parameters for the model."""
        self.γp = params["γp"]
        self.σp = params["σp"]
        self.γa = params["γa"]
        self.h2 = params["h2"]
        self.σeps = params["σeps"]
        self.f0 = params["f0"]
        self.EFAC = params["EFAC"]
        self.EQUAD = params["EQUAD"]

    @staticmethod
    def _compute_F_block(γ: float, dt: float) -> np.ndarray:
        """Compute the 2x2 state–transition matrix for the (r, a) block and the spin block.

        Parameters
        ----------
        γ : float
            Damping rate.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            2x2 state–transition matrix.

        """
        exp_term = np.exp(-γ * dt)
        return np.array(
            [
                [1.0, (1 - exp_term) / γ],
                [0.0, exp_term],
            ]
        )
    
    @profile
    def F_matrix(self, dt: float) -> np.ndarray:
        """Return the state–transition matrix for time step dt."""
        F_gw, F_spin, F_timing = get_F(self.γa, self.γp, dt, self.Npsr, self.M_sum)
        return F_gw, F_spin,F_timing

    @staticmethod
    def _compute_Q_block(γ: float, dt: float) -> np.ndarray:
        """Compute the 2x2 state–transition matrix for the (r, a) block and the spin block.

        Parameters
        ----------
        γ : float
            Damping rate.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            2x2 state–transition matrix.

        """
        exp_term = np.exp(-γ * dt)
        exp_2term = np.exp(-2 * γ * dt)

        q11 = (dt - 2 * (1 - exp_term) / γ + (1 - exp_2term) / (2 * γ)) / γ**3
        q12 = ((1 - exp_term) - (1 - exp_2term) / 2) / (γ**2)
        q22 = (1 - exp_2term) / (2 * γ)

        return np.array([[q11, q12], [q12, q22]])
    @profile
    def Q_matrix(self, dt: float) -> np.ndarray:
        """Return the process–noise covariance matrix for time step dt."""
        Q_gw, Q_spin, Q_timing = get_Q(self.γa, self.γp, dt, self.Npsr, self.M_sum, self.σeps)
        return Q_gw, Q_spin, Q_timing
    @profile
    def H_matrix(self, psr_idx: int) -> np.ndarray:
        """
        Return the observation matrix H for a given pulsar.

        For pulsar n the measurement equation is:
            δt = (1/f₀)·δφ − r + (design row)·[δε],
        so that H^(n) is the row vector:
            [1/f₀, 0, -1, 0, zeros(M[n])].

        Parameters
        ----------
        psr_idx : int
            Index of the pulsar.

        Returns
        -------
        np.ndarray
            Observation matrix (a row vector of length nx) for the specified pulsar.
        """
        # initialization
        H = np.zeros((1, self.nx))

        # update GW term
        H[0, 2 * psr_idx] = -1.0

        # update spin term
        H[0, self.Npsr * 2 + 2 * psr_idx] = 1.0 / self.f0[psr_idx]

        # # update timing term
        # # Use a precomputed start index
        # start_idx = self.M_start_indices[psr_idx]
        # end_idx = self.M_start_indices[psr_idx + 1]
        # row_idx = int(self.design_matrix_counter[psr_idx])
        # H[0, start_idx:end_idx] = self.pulsar_design_matrices[psr_idx][
        #     row_idx, :
        # ]  # length M[psr_idx]

        # # increment the counter for this pulsar
        # self.design_matrix_counter[psr_idx] += 1 # I think this will need to be moved for if we are doing repeated calls of the likelihood function 

        return H

    def R_matrix(self, σ, psr_idx: int) -> np.ndarray:
        """Build the measurement–noise covariance matrix R for the pulsars observed at a given epoch.

        For pulsar n, the measurement noise variance is (σt[n])².
        Currently, this method returns a scalar
        or a per-pulsar value.

        Returns
        -------
        Any
            The measurement noise covariance (for now, simply σt²).

        """
        return (σ * self.EFAC[psr_idx]) ** 2 + self.EQUAD[psr_idx] ** 2
