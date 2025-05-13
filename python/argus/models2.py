"""Module for specifying models to be used with a Kalman filter."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List
from python.argus.model import get_F, get_Q
import sys 
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


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

    When use_gw=False, the GW term (-r) is removed from the measurement equation.
    """

    def __init__(
        self,
        df_psr: Any,
        hd_correlation_matrix: np.ndarray,
        pulsar_design_matrices: np.ndarray,
        use_gw: bool = True
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
        pulsar_design_matrices : np.ndarray
            Design matrices for each pulsar
        use_gw : bool, optional
            If True, include GW terms in the measurement equation. If False, 
            use null model (GW states still present but not used in measurements).
            Default is True.
        """
        self.Npsr = int(len(df_psr))
        print("The number of pulsars is:", self.Npsr)
        self.name = "Stochastic GW background model"
        self.use_gw = use_gw
        
        if not self.use_gw:
            logger.info("Initializing null GW model - GW states present but not used in measurements")
        
        # Total state dimension: for each pulsar, two state variables from spin noise,
        # two from GW noise, and dim_M extra parameters.
        self.nx = self.Npsr * (2 + 2) + df_psr["dim_M"].sum()

        self.M = df_psr["dim_M"].values.astype(int)  # array of integers
        self.M_sum = self.M.sum()

        self.hd_correlation_matrix = hd_correlation_matrix

        self.M_start_indices = np.cumsum([0] + [m for m in self.M]) + 4 * self.Npsr

        self.f0 = df_psr["F0"].values

        print("The frequencies are:", self.f0)

        # Used in the H_matrix function
        self.pulsar_design_matrices = pulsar_design_matrices
        self.design_matrix_counter = np.zeros(self.Npsr)
        self.F = None


    def F_matrix(self, dt: float) -> np.ndarray:
        """Return the state–transition matrix for time step dt."""
        F_gw, F_spin = get_F(self.γa, self.γp, dt, self.Npsr, self.M_sum)
        return F_gw, F_spin

    def Q_matrix(self, dt: float) -> np.ndarray:
        """Return the process–noise covariance matrix for time step dt."""
        Q_gw, Q_spin, Q_timing = get_Q(self.γa, self.γp, dt, self.Npsr, self.M_sum, self.σeps)
        return Q_gw, Q_spin, Q_timing



    def H_matrix(self, t_idx: int) -> np.ndarray: 
        """At timestep t_idx, get the correct H-matrix."""
        return self.H_matrix_list[t_idx]
    
    def compute_H_matrix_for_step(self, time_step_index: int) -> np.ndarray:
        """
        Compute the observation matrix H for the current time step using NumPy.

        This matrix maps the full state vector to the vector of observations
        from all pulsars at the given time step.

        Parameters
        ----------
        time_step_index : int
            The index corresponding to the current observation time step.

        Returns
        -------
        np.ndarray
            Observation matrix H of shape (Npsr, nx) for the current step.
        """
        # Initialize the H matrix with zeros using NumPy
        H = np.zeros((self.Npsr, self.nx))

        # Loop over each pulsar to build the corresponding row of H
        for psr_idx in range(self.Npsr):
            # Indices in the state vector 'x' relevant to this pulsar (psr_idx)
            redshift_idx = 2 * psr_idx
            spin_idx = self.Npsr * 2 + 2 * psr_idx
            tm_start_idx = self.M_start_indices[psr_idx]
            tm_end_idx = self.M_start_indices[psr_idx + 1]

            # Get the relevant row from this pulsar's precomputed design matrix
            design_row = self.pulsar_design_matrices[psr_idx][time_step_index, :]

            # Update Redshift term coefficient (-1.0) only if use_gw is True
            if self.use_gw:
                H[psr_idx, redshift_idx] = -1.0

            # Update Spin noise term coefficient (1 / f0_n)
            H[psr_idx, spin_idx] = 1.0 / self.f0[psr_idx]

            # Update Timing model term coefficients (design matrix row)
            H[psr_idx, tm_start_idx:tm_end_idx] = design_row

        return H

    def compute_all_H_matrices(self) -> List[np.ndarray]:
        """
        Compute the observation matrix H for all time steps using NumPy.

        This iterates through all time steps, calling compute_H_matrix_for_step
        for each one, using the appropriate row from the time-varying design matrices.

        Assumptions are the same as the JAX version but using NumPy arrays.

        Returns
        -------
        List[np.ndarray]
            A list where each element is the NumPy H matrix (Npsr, nx)
            for a specific time step, ordered from time step 0 onwards.
        """
        if not self.pulsar_design_matrices or self.Npsr == 0:
            print("Warning: No pulsars or design matrices found. Returning empty list.")
            return []

        try:
            num_time_steps = self.pulsar_design_matrices[0].shape[0]
        except (IndexError, AttributeError):
             raise ValueError("Cannot determine number of time steps. "
                              "Ensure self.pulsar_design_matrices is a list of NumPy arrays "
                              "with at least one element.")

        # Optional consistency check (same as before)
        for i in range(1, self.Npsr):
             if self.pulsar_design_matrices[i].shape[0] != num_time_steps:
                 raise ValueError(f"Inconsistent number of time steps found (Pulsar 0: {num_time_steps}, Pulsar {i}: {self.pulsar_design_matrices[i].shape[0]})")

        print(f"Computing H matrices for all {num_time_steps} time steps (using NumPy)...")

        all_H = []
        for t_idx in range(num_time_steps):
            H_step = self.compute_H_matrix_for_step(t_idx)
            all_H.append(H_step)

        print("Finished computing all H matrices.")
        return all_H

    def precompute_H_matrix(self) -> np.ndarray:
        """
        Compute H for all steps using NumPy and stacks them into a single 3D array.

        See compute_all_H_matrices for assumptions.

        Returns
        -------
        np.ndarray
            A single NumPy array of shape (num_time_steps, Npsr, nx).
            Returns an empty array with correct dimensions if no steps exist.
        """
        list_of_H = self.compute_all_H_matrices() # Calls the list-based version

        if not list_of_H:
            try:
                num_time_steps = self.pulsar_design_matrices[0].shape[0] if self.pulsar_design_matrices else 0
            except (IndexError, AttributeError):
                num_time_steps = 0
            # Return empty NumPy array with correct shape
            return np.zeros((num_time_steps, self.Npsr, self.nx))

        # Stack the list of 2D arrays along a new axis (axis 0) using NumPy
        return np.stack(list_of_H, axis=0)




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
        sys.exit()
        return (σ * self.EFAC[psr_idx]) ** 2 + self.EQUAD[psr_idx] ** 2
