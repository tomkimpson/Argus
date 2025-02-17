"""Module for specifying models to be used with a Kalman filter."""

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import block_diag
from typing import Any, Dict, List


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

    The state vector for pulsar n is assumed to be:

        X^(n) = [δφ, δf, r, a, δε₁, ..., δε_M]^T

    with measurement equation

        δt^(n) = (1/f₀)·δφ − r + (design row)·[δε].

    The measurement row for pulsar n is therefore

        [1/f₀, 0, -1, 0, zeros(M[n])],

    where M[n] is the number of extra (design) parameters for that pulsar.
    """

    def __init__(self, df_psr: Any) -> None:
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

        """
        self.Npsr = len(df_psr)
        print("The number of pulsars is:", self.Npsr)
        self.name = "Stochastic GW background model"
        # Total state dimension: for each pulsar, two state variables from spin noise,
        # two from GW noise, and dim_M extra parameters.
        self.nx = self.Npsr * (2 + 2) + df_psr["dim_M"].sum()
        self.M = df_psr["dim_M"].values.astype(int)

    def set_global_parameters(self, params: Dict[str, Any]) -> None:
        """Set global parameters for the model.

        The params dictionary must include:
            "γp" : np.ndarray
                Array of spin–noise damping rates for each pulsar.
            "σp" : np.ndarray
                Array of spin–noise white noise amplitudes for each pulsar.
            "γa" : float
                GW damping rate.
            "h2" : float
                Mean–square GW strain (<h²>).
            "σeps" : float
                White–noise amplitude for timing model parameters.
            "separation_angle_matrix": np.ndarray
                (N x N) symmetric array of angular separations (radians)
                between pulsars.
            "f0" : np.ndarray
                Array of pulsar spin frequencies.
            "σt" : np.ndarray or float
                Measurement noise standard deviation (per pulsar).

        Note:
        ----
        Although "Delta_t" is mentioned in some documentation, here dt is passed
        directly to F_matrix and Q_matrix.

        """
        self.γp = params["γp"]  # shape: (Npsr,)
        self.σp = params["σp"]  # shape: (Npsr,)
        self.γa = params["γa"]  # scalar
        self.h2 = params["h2"]  # scalar
        self.σeps = params["σeps"]  # scalar (could be extended per pulsar)
        self.separation_angle_matrix = params["separation_angle_matrix"]
        self.f0 = params["f0"]
        self.σt = params["σt"]

    @staticmethod
    def _hellings_downs(θ: np.ndarray) -> np.ndarray:
        """Compute the Hellings–Downs function for an angle θ (in radians).

        For θ == 0 the autocorrelation is defined to be 1.

        Parameters
        ----------
        θ : np.ndarray or float
            Angular separation(s) in radians.

        Returns
        -------
        np.ndarray
            Hellings–Downs values.

        Notes
        -----
        To avoid computing log(0), np.where is used to set the value to 1 when
        θ is 0.

        """
        x = (1 - np.cos(θ)) / 2.0
        return np.where(
            np.isclose(θ, 0.0),
            1.0,
            (3 / 2) * x * np.log(x) - x / 4 + 0.5,
        )

    @staticmethod
    def _compute_F_p(γp: float, dt: float) -> np.ndarray:
        """Compute the 2x2 state–transition matrix for the (δφ, δf) block.

        Parameters
        ----------
        γp : float
            Spin–noise damping rate.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            2x2 state–transition matrix.

        """
        exp_term = np.exp(-γp * dt)
        return np.array([
            [1.0, (1 - exp_term) / γp],
            [0.0, exp_term],
        ])

    @staticmethod
    def _compute_F_a(γa: float, dt: float) -> np.ndarray:
        """Compute the 2x2 state–transition matrix for the (r, a) block.

        Parameters
        ----------
        γa : float
            GW damping rate.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            2x2 state–transition matrix.

        """
        exp_term = np.exp(-γa * dt)
        return np.array([
            [1.0, (1 - exp_term) / γa],
            [0.0, exp_term],
        ])

    @staticmethod
    def _compute_Q_block(γ: float, σ: float, dt: float) -> np.ndarray:
        """Compute the 2x2 discretized noise covariance for a process with damping rate γ and white noise amplitude σ.

        Parameters
        ----------
        γ : float
            Damping rate.
        σ : float
            White noise amplitude.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            2x2 noise covariance matrix.

        """
        exp_term = np.exp(-γ * dt)
        exp_2γ_dt = np.exp(-2 * γ * dt)
        term1 = (dt / γ**2 - 2 * (1 - exp_term) / γ**3 +
                 (1 - exp_2γ_dt) / (2 * γ**3))
        term2 = (1 - exp_term) / γ**2 - (1 - exp_2γ_dt) / (2 * γ**2)
        term3 = (1 - exp_2γ_dt) / (2 * γ)
        return σ**2 * np.array([
            [term1, term2],
            [term2, term3],
        ])

    def F_matrix(self, dt: float) -> np.ndarray:
        """Build the overall state–transition matrix F (block–diagonal over pulsars).

        For pulsar n, the state block is:

              [ F_p^(n)    0       0 ]
              [    0      F_a     0 ]
              [    0       0    I_(M[n]) ]

        where F_p^(n) is 2x2, F_a is 2x2 (common to all pulsars), and
        I_(M[n]) is an identity matrix of size M[n].

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Block–diagonal state–transition matrix.

        """
        F_a = self._compute_F_a(self.γa, dt)
        F_blocks = [
            block_diag(
                self._compute_F_p(self.γp[i], dt),
                F_a,
                np.eye(self.M[i]),
            )
            for i in range(self.Npsr)
        ]
        return block_diag(*F_blocks)

    def Q_matrix(self, dt: float) -> np.ndarray:
        """Build the full process–noise covariance matrix Q.

        For each pulsar n the diagonal block is:
            Q^(n,n) = block_diag(Q_p^(n), Q_a^(n,n), Q_ε^(n))
        where:
          - Q_p^(n) is the 2x2 spin–noise covariance (using γp and σp).
          - Q_a^(n,n) is a 2x2 GW noise covariance with
              σ_a^(n,n) = sqrt((h2/6) * γa)
            (using Γ = 1 for the autocorrelation).
          - Q_ε^(n) is the timing–model covariance:
              σeps² · dt · I_(M[n]).

        Off–diagonal blocks (n ≠ m) are zero except in the GW sector. For
        pulsars n and m one uses:
              σ_a^(n,m)² = (h2/6) * γa * Γ(θ_nm),
        placing the corresponding 2x2 covariance in the GW blocks.

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Process–noise covariance matrix.

        """
        block_sizes = [2 + 2 + self.M[i] for i in range(self.Npsr)]
        cum_sizes = np.concatenate(([0], np.cumsum(block_sizes)))
        total_size = cum_sizes[-1]
        Q = np.zeros((total_size, total_size))
        Q_a_template = lambda σ: self._compute_Q_block(self.γa, σ, dt)

        for n in range(self.Npsr):
            i0, i1 = cum_sizes[n], cum_sizes[n + 1]
            Q_p = self._compute_Q_block(self.γp[n], self.σp[n], dt)
            σ_a_nn = np.sqrt((self.h2 / 6) * self.γa)
            Q_a_nn = Q_a_template(σ_a_nn)
            M_dim = self.M[n]
            Q_ε = self.σeps**2 * dt * np.eye(M_dim)
            Q_block = block_diag(Q_p, Q_a_nn, Q_ε)
            Q[i0:i1, i0:i1] = Q_block

        for n in range(self.Npsr):
            for m in range(n + 1, self.Npsr):
                θ_nm = self.separation_angle_matrix[n, m]
                Γ_nm = self._hellings_downs(θ_nm)
                σ_a_nm = np.sqrt((self.h2 / 6) * self.γa * Γ_nm)
                Q_a_nm = Q_a_template(σ_a_nm)
                i0_n = cum_sizes[n] + 2
                i1_n = i0_n + 2
                i0_m = cum_sizes[m] + 2
                i1_m = i0_m + 2
                Q[i0_n:i1_n, i0_m:i1_m] = Q_a_nm
                Q[i0_m:i1_m, i0_n:i1_n] = Q_a_nm.T

        return Q

    # def _compute_H_matrix_list(self) -> List[np.ndarray]:
    #     """Build a list of measurement matrices H (one per pulsar).

    #     For pulsar n the measurement equation is:

    #         δt = (1/f₀)·δφ − r + (design row)·[δε],

    #     so that H^(n) is the row vector:

    #         [1/f₀, 0, -1, 0, zeros(M[n])].

    #     The design row (beyond the first four elements) is assumed to be zero.

    #     Returns
    #     -------
    #     List[np.ndarray]
    #         A list of 1 x (4 + M[n]) arrays (one per pulsar).

    #     """
    #     H_list = [
    #         np.concatenate(
    #             (
    #                 np.array([1.0 / self.f0[i], 0.0, -1.0, 0.0]),
    #                 np.zeros(self.M[i]),
    #             )
    #         ).flatten()
    #         for i in range(self.Npsr)
    #     ]
    #     return H_list
    

    def H_matrix(self,psr_idx: int)-> np.ndarray:
        
        output = np.zeros(self.nx) 

        # 1. Compute the start index for segment i
        #    Each segment j has length (4 + M[j]).
        #    So, to find the start of segment i, sum all lengths up to i.
        start_idx = sum(4 + self.M[j] for j in range(psr_idx)) #Compute the start index for the ith segement

        # 2. Construct the i-th segment
        #    It's basically [1/f0[i], 0, -1, 0] concatenated with zeros(M[i]).
        segment = np.concatenate((
            np.array([1.0 / self.f0[psr_idx], 0.0, -1.0, 0.0]),
            np.zeros(self.M[psr_idx])
        ))
        seg_len = len(segment)  # This should be 4 + M[i]


        output[start_idx : start_idx + seg_len] = segment

        
        return output


    def R_matrix(self) -> Any:
        """Build the measurement–noise covariance matrix R for the pulsars observed at a given epoch.

        For pulsar n, the measurement noise variance is (σt[n])².
        Currently, this method returns a scalar (if σt is the same for all pulsars)
        or a per-pulsar value.

        Returns
        -------
        Any
            The measurement noise covariance (for now, simply σt²).
            
        """
        return self.σt**2
    
