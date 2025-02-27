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

    def __init__(self, df_psr: Any, hd_correlation_matrix: np.ndarray) -> None:
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
        self.M = df_psr["dim_M"].values.astype(int)
        self.hd_correlation_matrix = hd_correlation_matrix

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

        # Precompute dimensions and indices
        N = self.Npsr
        self.block_sizes = np.array([3 + m for m in self.M])
        self.cum_sizes = np.concatenate(([0], np.cumsum(self.block_sizes)))
        self.total_size = self.cum_sizes[-1] + N
        
        # Precompute indices for matrix blocks
        self.block_starts = self.cum_sizes[:-1] + N
        self.r_indices = self.block_starts + 2
        self.diag_indices = self.block_starts + 1
        self.col_indices = self.block_starts + 1
        self.row_indices = self.block_starts
        
        # Precompute common terms for matrices
        self.Q_a_base = (self.h2 / 6) * self.γa * self.hd_correlation_matrix
        
        # Precompute timing parameter indices for vectorized operations
        self.timing_param_indices = np.concatenate([
            np.arange(start + 3, start + 3 + m) 
            for start, m in zip(self.block_starts, self.M)
        ])

    def F_matrix(self, dt: float) -> np.ndarray:
        """Return the state–transition matrix for time step dt."""
        # Initialize with identity
        F = np.eye(self.total_size)
        
        # 1. F_aa block (top-left)
        F[:self.Npsr, :self.Npsr] *= np.exp(-self.γa * dt)
        
        # 2. F_xx block - compute and place all exponential terms at once
        exp_terms = np.exp(-self.γp * dt)
        F[self.row_indices, self.col_indices] = (1 - exp_terms) / self.γp
        F[self.diag_indices, self.diag_indices] = exp_terms
        
        # 3. F_xa block (bottom-left)
        coupling_term = (1 - np.exp(-self.γa * dt)) / self.γa
        F[self.r_indices[:, np.newaxis], np.arange(self.Npsr)] = coupling_term
        
        return F

    def _compute_I_integrals(self, γp: float, dt: float) -> tuple:
        """Compute the I1, I2, I3 integrals for a given γp and dt.
        
        Parameters
        ----------
        γp : float
            Damping rate
        dt : float
            Time step

        Returns
        -------
        tuple
            (I1, I2, I3) values
        """
        exp_term = np.exp(-γp * dt)
        exp_2term = np.exp(-2 * γp * dt)
        
        # I3 = (1 - exp(-2γp·dt))/(2γp)
        I3 = (1 - exp_2term) / (2 * γp)
        
        # I2 = [2(1 - exp(-γp·dt)) - (1 - exp(-2γp·dt))]/(2γp²)
        I2 = (2 * (1 - exp_term) - (1 - exp_2term)) / (2 * γp**2)
        
        # I1 = [dt - 2(1 - exp(-γp·dt))/γp + (1 - exp(-2γp·dt))/(2γp)]/γp²
        I1 = (dt - 2 * (1 - exp_term) / γp + (1 - exp_2term) / (2 * γp)) / γp**2
        
        return I1, I2, I3

    def _compute_I_integrals_vectorized(self, γp: np.ndarray, dt: float) -> tuple:
        """Compute the I1, I2, I3 integrals for all γp values at once."""
        exp_term = np.exp(-γp * dt)
        exp_2term = np.exp(-2 * γp * dt)
        
        I3 = (1 - exp_2term) / (2 * γp)
        I2 = (2 * (1 - exp_term) - (1 - exp_2term)) / (2 * γp**2)
        I1 = (dt - 2 * (1 - exp_term) / γp + (1 - exp_2term) / (2 * γp)) / γp**2
        
        return I1, I2, I3

    def Q_matrix(self, dt: float) -> np.ndarray:
        """Return the process–noise covariance matrix for time step dt."""
        N = self.Npsr
        Q = np.zeros((self.total_size, self.total_size))
        
        # 1. Build Q_aa block (top-left)
        aa_factor = (1 - np.exp(-2 * self.γa * dt)) / (2 * self.γa)
        Q_a = aa_factor * self.Q_a_base
        Q[:N, :N] = Q_a
        
        # 2. Build Q_xx blocks - vectorized approach
        # Compute all I-integrals at once
        I1, I2, I3 = self._compute_I_integrals_vectorized(self.γp, dt)
        
        # Place all (δφ,δf) blocks at once
        for i in range(2):
            for j in range(2):
                if i == 0 and j == 0:
                    values = I1 * self.σp**2
                elif i == 1 and j == 1:
                    values = I3 * self.σp**2
                else:
                    values = I2 * self.σp**2
                
                rows = self.block_starts + i
                cols = self.block_starts + j
                Q[rows, cols] = values
        
        # Add timing parameter variances (diagonal terms) - vectorized
        Q[self.timing_param_indices, self.timing_param_indices] = self.σeps**2 * dt
        
        # 3. Build Q_xa blocks using F_xa coupling
        coupling_term = (1 - np.exp(-self.γa * dt)) / self.γa
        
        # Compute Q_xa and Q_ax
        for i in range(N):
            for j in range(N):
                Q[self.r_indices[i], j] = coupling_term * Q_a[i, j]
                Q[j, self.r_indices[i]] = coupling_term * Q_a[i, j]
        
        return Q

    def H_matrix(self, psr_idx: int) -> np.ndarray:
        """Build a list of measurement matrices H (one per pulsar).

        For pulsar n the measurement equation is:

            δt = (1/f₀)·δφ − r + (design row)·[δε],

        so that H^(n) is the row vector:

            [1/f₀, 0, -1, 0, zeros(M[n])].

        The design row (beyond the first four elements) is assumed to be zero.

        Returns
        -------
        List[np.ndarray]
            A list of 1 x (4 + M[n]) arrays (one per pulsar).

        """
        output = np.zeros(self.nx)

        # 1. Compute the start index for segment i
        #    Each segment j has length (4 + M[j]).
        #    So, to find the start of segment i, sum all lengths up to i.
        start_idx = sum(
            4 + self.M[j] for j in range(psr_idx)
        )  # Compute the start index for the ith segement

        # 2. Construct the i-th segment
        #    It's basically [1/f0[i], 0, -1, 0] concatenated with zeros(M[i]).
        segment = np.concatenate(
            (
                np.array([1.0 / self.f0[psr_idx], 0.0, -1.0, 0.0]),
                np.zeros(self.M[psr_idx]),
            )
        )
        seg_len = len(segment)  # This should be 4 + M[i]

        output[start_idx : start_idx + seg_len] = segment

        return output

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


class SGWB_restructured(ModelHyperClass):
    """A restructured model class for the Stochastic Gravitational Wave Background.
    
    The state vector for pulsar n is assumed to be:
        X^(n) = [δφ, δf, r, a, δε₁, ..., δε_M]^T
    
    with measurement equation
        δt^(n) = (1/f₀)·δφ − r + (design row)·[δε].
    """

    def __init__(self, df_psr: Any, hd_correlation_matrix: np.ndarray) -> None:
        """Initialize the SGWB_restructured model.

        Parameters
        ----------
        df_psr : DataFrame
            A pandas DataFrame containing pulsar information. The DataFrame is
            assumed to contain at least the following columns:
                - dim_M: integer, the number of design parameters for that pulsar.
                - f0: the pulsar spin frequency.
                - sigma_t: the measurement noise standard deviation.
        hd_correlation_matrix : np.ndarray
            Precomputed Hellings-Downs correlation matrix
        """
        self.Npsr = len(df_psr)
        print("The number of pulsars is:", self.Npsr)
        self.name = "Restructured Stochastic GW background model"
        # Total state dimension: for each pulsar, two state variables from spin noise,
        # two from GW noise, and dim_M extra parameters.
        self.nx = self.Npsr * (2 + 2) + df_psr["dim_M"].sum()
        self.M = df_psr["dim_M"].values.astype(int)
        self.hd_correlation_matrix = hd_correlation_matrix

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

        # Precompute dimensions and indices
        N = self.Npsr
        self.block_sizes = np.array([3 + m for m in self.M])
        self.cum_sizes = np.concatenate(([0], np.cumsum(self.block_sizes)))
        self.total_size = self.cum_sizes[-1] + N
        
        # Precompute indices for matrix blocks
        self.block_starts = self.cum_sizes[:-1] + N
        self.r_indices = self.block_starts + 2
        self.diag_indices = self.block_starts + 1
        self.col_indices = self.block_starts + 1
        self.row_indices = self.block_starts
        
        # Precompute common terms for matrices
        self.Q_a_base = (self.h2 / 6) * self.γa * self.hd_correlation_matrix
        
        # Precompute timing parameter indices for vectorized operations
        self.timing_param_indices = np.concatenate([
            np.arange(start + 3, start + 3 + m) 
            for start, m in zip(self.block_starts, self.M)
        ])

    def F_matrix(self, dt: float) -> np.ndarray:
        """Return the state–transition matrix for time step dt."""
        # Initialize with identity
        F = np.eye(self.total_size)
        
        # 1. F_aa block (top-left)
        F[:self.Npsr, :self.Npsr] *= np.exp(-self.γa * dt)
        
        # 2. F_xx block - compute and place all exponential terms at once
        exp_terms = np.exp(-self.γp * dt)
        F[self.row_indices, self.col_indices] = (1 - exp_terms) / self.γp
        F[self.diag_indices, self.diag_indices] = exp_terms
        
        # 3. F_xa block (bottom-left)
        coupling_term = (1 - np.exp(-self.γa * dt)) / self.γa
        F[self.r_indices[:, np.newaxis], np.arange(self.Npsr)] = coupling_term
        
        return F

    def Q_matrix(self, dt: float) -> np.ndarray:
        """Return the process–noise covariance matrix for time step dt."""
        N = self.Npsr
        Q = np.zeros((self.total_size, self.total_size))
        
        # 1. Build Q_aa block (top-left)
        aa_factor = (1 - np.exp(-2 * self.γa * dt)) / (2 * self.γa)
        Q_a = aa_factor * self.Q_a_base
        Q[:N, :N] = Q_a
        
        # 2. Build Q_xx blocks - vectorized approach
        # Compute all I-integrals at once
        I1, I2, I3 = self._compute_I_integrals_vectorized(self.γp, dt)
        
        # Place all (δφ,δf) blocks at once
        for i in range(2):
            for j in range(2):
                if i == 0 and j == 0:
                    values = I1 * self.σp**2
                elif i == 1 and j == 1:
                    values = I3 * self.σp**2
                else:
                    values = I2 * self.σp**2
                
                rows = self.block_starts + i
                cols = self.block_starts + j
                Q[rows, cols] = values
        
        # Add timing parameter variances (diagonal terms) - vectorized
        Q[self.timing_param_indices, self.timing_param_indices] = self.σeps**2 * dt
        
        # 3. Build Q_xa blocks using F_xa coupling
        coupling_term = (1 - np.exp(-self.γa * dt)) / self.γa
        
        # Compute Q_xa and Q_ax
        for i in range(N):
            for j in range(N):
                Q[self.r_indices[i], j] = coupling_term * Q_a[i, j]
                Q[j, self.r_indices[i]] = coupling_term * Q_a[i, j]
        
        return Q

    def H_matrix(self, psr_idx: int) -> np.ndarray:
        """Return the observation matrix for a given pulsar.

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
            Observation matrix for the specified pulsar.
        """
        output = np.zeros(self.nx)
        start_idx = sum(4 + self.M[j] for j in range(psr_idx))
        segment = np.concatenate(
            (
                np.array([1.0 / self.f0[psr_idx], 0.0, -1.0, 0.0]),
                np.zeros(self.M[psr_idx]),
            )
        )
        seg_len = len(segment)
        output[start_idx : start_idx + seg_len] = segment
        return output

    def R_matrix(self, σ: float, psr_idx: int) -> np.ndarray:
        """Return the observation noise covariance for a given pulsar.

        Parameters
        ----------
        σ : float
            Measurement error.
        psr_idx : int
            Index of the pulsar.

        Returns
        -------
        float
            Observation noise variance.
        """
        return (σ * self.EFAC[psr_idx]) ** 2 + self.EQUAD[psr_idx] ** 2

    def _compute_I_integrals_vectorized(self, γp: np.ndarray, dt: float) -> tuple:
        """Compute the I1, I2, I3 integrals for all γp values at once."""
        exp_term = np.exp(-γp * dt)
        exp_2term = np.exp(-2 * γp * dt)
        
        I3 = (1 - exp_2term) / (2 * γp)
        I2 = (2 * (1 - exp_term) - (1 - exp_2term)) / (2 * γp**2)
        I1 = (dt - 2 * (1 - exp_term) / γp + (1 - exp_2term) / (2 * γp)) / γp**2
        
        return I1, I2, I3

    def profile_performance(self, dt: float, n_iterations: int = 100):
        """Profile the performance of F_matrix and Q_matrix.
        
        Parameters
        ----------
        dt : float
            Time step to use for profiling
        n_iterations : int
            Number of iterations to run
            
        Returns
        -------
        dict
            Dictionary with profiling results
        """
        import time
        import cProfile
        import pstats
        from io import StringIO
        
        results = {}
        
        # Time F_matrix
        start = time.time()
        for _ in range(n_iterations):
            F = self.F_matrix(dt)
        end = time.time()
        results['F_matrix_time'] = (end - start) / n_iterations
        
        # Time Q_matrix
        start = time.time()
        for _ in range(n_iterations):
            Q = self.Q_matrix(dt)
        end = time.time()
        results['Q_matrix_time'] = (end - start) / n_iterations
        
        # Profile F_matrix
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(10):  # Fewer iterations for profiling
            F = self.F_matrix(dt)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        results['F_matrix_profile'] = s.getvalue()
        
        # Profile Q_matrix
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(10):  # Fewer iterations for profiling
            Q = self.Q_matrix(dt)
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        results['Q_matrix_profile'] = s.getvalue()
        
        return results
