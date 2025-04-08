"""Module which implements Kalman filter algorithm."""

from typing import Tuple, NamedTuple
import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
from argus.models import ModelHyperClass
from argus.types import Array
from argus.jmath import get_xp, get_Pp, update_x_P

from line_profiler import profile

class ScalarKalmanFilter:
    """A class to implement the linear Kalman filter on scalar inputs.

    It takes four initialization arguments:

        `Model`: instance of `modelHyperClass` class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc.

        `Observations`: 2D array which holds the noisy observations recorded at the detector

        `x0`: instance of `StateData` which holds the initial guess of the initial states

        `P0`: instance of `CovarianceData` which holds the uncertainty in the guess of P0

    ...and a placeholder **kwargs, which is not currently used.
    """

    def __init__(self, model: ModelHyperClass, observations: Array, x0: NamedTuple, P0: NamedTuple, **kwargs):
        """Initialize the class."""
        self.model = model
        self.observations = observations
        self.x0 = self._unpack_state(x0)
        self.P0 = self._unpack_covariance(P0)
       
        # Extract the observations into separate arrays
        self.toa = self.observations[:, 0]
        self.data = self.observations[:, 1]
        self.data_errors = self.observations[:, 2]
        self.psr_indices = self.observations[:, 3].astype(int)
        self.N_timesteps = len(self.observations)
        self.t_diffs = jnp.diff(self.toa)

        assert np.isscalar(self.data[0])

        # pre-calculate H
        self.H_eps_full = self.model.H_matrix_full_list(self.psr_indices)

    def _unpack_state(self, x: NamedTuple):
        """unpack namedtuple state to tuple (for faster execution)"""
        return x.gw, x.spin, x.eps
    
    def _unpack_covariance(self, P: NamedTuple):
        """unpack namedtuple covariance to tuple (for faster execution)"""
        return P.gw, P.spin, P.eps, P.gw_spin, P.gw_eps, P.spin_eps

    def _log_likelihood(self, y, cov):
        """Given the innovation and innovation covariance, get the likelihood."""
        log_likelihood = -0.5 * (np.log(2.0 * np.pi * cov) + (y * y) / cov)
        return log_likelihood

    @profile
    def predict(self, dt: Array, x: Tuple, P: Tuple):
        # Define the F matrices for each block
        F = self.model.F_matrix(dt) # (F_gw, F_spin)
        # jax.block_until_ready(F)
        # Define the Q matrices for each block
        Q = self.model.Q_matrix(dt) # (Q_gw, Q_spin, Q_eps)
        # jax.block_until_ready(Q)
        # Predict the next state
        xp = get_xp(F, x)
        # jax.block_until_ready(xp)
        # Predict the next covariance
        Pp = get_Pp(F, P, Q)
        # jax.block_until_ready(Pp)
        return xp, Pp

    @profile
    def update(self, psr_index: Array, x: Tuple, P: Tuple, H_eps: Array, R: Array, y: Array):
        """
        Perform one Kalman update for a single scalar measurement y

        Returns
        -------
        Updated (x_gw, x_spin, x_eps, P_gw, P_gw_spin, P_gw_eps, P_spin, P_spin_eps, P_eps).
        """
        f0 = self.model.f0[psr_index]
        
        x_up, P_up, ll_t = update_x_P(x, P, psr_index, f0, H_eps, R, y)
        # jax.block_until_ready(x_up)
        self.ll += ll_t
        
        return x_up, P_up
        

        
    @profile
    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        # Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
        self.model.set_global_parameters(θ)

        #Initialise the likelihood
        self.ll = 0.0
        
        #Initialise the index
        i = 0
        # Update step
        x_up, P_up = self.update(
            psr_index=int(self.psr_indices[i]),
            x=self.x0,
            P=self.P0,
            H_eps=self.H_eps_full[i],
            R=self.data_errors[i],
            y=self.data[i]
        )
        
        # Iterate over the data.
        # tqdm progress bar just for testing. To be removed ultimately
        for i in tqdm(range(1, self.N_timesteps), desc="Processing timesteps"):
            # Get time step
            dt = self.t_diffs[i - 1]

            # Predict step
            xp, Pp = self.predict(dt=dt, x=x_up, P=P_up)
            
            # Update step
            x_up, P_up = self.update(
                psr_index=int(self.psr_indices[i]),
                x = xp,
                P = Pp,
                H_eps       = self.H_eps_full[i],
                R          = self.data_errors[i],
                y          = self.data[i]
            )

        return self.ll
