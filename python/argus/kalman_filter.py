"""Module which implements Kalman filter algorithm."""

import numpy as np
from tqdm import tqdm
from argus.jmath import get_Pp
from line_profiler import profile

class ScalarKalmanFilter:
    """A class to implement the linear Kalman filter on scalar inputs.

    It takes four initialization arguments:

        `Model`: class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc.

        `Observations`: 2D array which holds the noisy observations recorded at the detector

        `x0`: A 1D array which holds the initial guess of the initial states

        `P0`: The uncertainty in the guess of P0

    ...and a placeholder **kwargs, which is not currently used.
    """

    def __init__(self, model, observations, x0, P0, **kwargs):
        """Initialize the class."""
        self.model = model
        self.observations = observations
        self.x0 = x0
        self.P0 = P0

        # Extract the observations into separate arrays
        self.toa = self.observations[:, 0]
        self.data = self.observations[:, 1]
        self.data_errors = self.observations[:, 2]
        self.psr_indices = self.observations[:, 3].astype(int)
        self.N_timesteps = len(self.observations)
        self.t_diffs = np.diff(self.toa)

        assert np.isscalar(self.data[0])

    def _log_likelihood(self, y, cov):
        """Given the innovation and innovation covariance, get the likelihood."""
        log_likelihood = -0.5 * (np.log(2.0 * np.pi * cov) + (y * y) / cov)

        if hasattr(log_likelihood, "item"):
            log_likelihood = log_likelihood.item()

        return log_likelihood
    @profile
    def predict(self, dt):
        """Predict the next state and covariance."""
        F = self.model.F_matrix(dt)
        Q = self.model.Q_matrix(dt)

        self.xp = F @ self.x
        # self.Pp = F @ self.P @ F.T + Q
        self.Pp = get_Pp(F, self.P, Q)
        # breakpoint()
    @profile
    def update(self, z, z_err, psr_index):
        """Update the state and covariance with a new observation."""
        # Define the time-dependent H and R matrices for this timestep
        self.H = self.model.H_matrix(psr_index)
        self.R = self.model.R_matrix(z_err, psr_index)

        # Now run through the update algorithm
        y = z - self.H @ self.xp  # innovation. For this class, this is just a scalar
        S = self.H @ self.Pp @ self.H.T + self.R  # innovation covariance, a scalar
        K = self.Pp @ self.H.T / S  # Kalman gain, dimension (n_x, 1)
        self.x = self.xp + K * y  # Updated state, dimension (n_x, 1)
        # breakpoint()
        self.P = (np.eye(len(self.xp)) - K @ self.H) @ self.Pp  # Updated covariance, dimension (n_x, n_x)
        
        # self.ll += self._log_likelihood(y, S)  # update the likelihood
    @profile
    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        # Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
        self.model.set_global_parameters(θ)

        # Initialise x and P, the likelihood, and the index i
        # self.x should be of dimension (n_x,1)
        self.x, self.P, self.ll, i = self.x0.reshape(-1, 1), self.P0, 0.0, int(0)

        # Do the first update step
        ##For the first update step, just assign the predicted values to be the states
        self.xp, self.Pp = self.x, self.P
        # breakpoint()
        ##Update step
        self.update(
            self.data[i], self.data_errors[i], self.psr_indices[i]
        )  # Updates x,P,and the likelihood_value

        # Iterate over the data.
        # tqdm progress bar just for testing. To be removed ultimately
        for i in tqdm(range(1, self.N_timesteps), desc="Processing timesteps"):
            # Set the delta t
            dt = self.t_diffs[i - 1]

            # Predict step
            self.predict(dt)

            # Update step
            self.update(
                self.data[i], self.data_errors[i], self.psr_indices[i]
            )  # Updates x,P,and the likelihood_value

        return self.ll
