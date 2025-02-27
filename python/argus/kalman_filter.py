"""Module which implements Kalman filter algorithm."""

import numpy as np
from tqdm import tqdm
import time 

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
        return -0.5 * (np.log(2.0 * np.pi * cov) + (y * y) / cov)

    def predict(self, dt):
        """Predict the next state and covariance."""
        start_time = time.time()
        F = self.model.F_matrix(dt)
        end_time = time.time()
        print(f"Time taken to get F matrix: {end_time - start_time:.4f} seconds")

        
        Q = self.model.Q_matrix(dt)
        end_time = time.time()
        print(f"Time taken to get Q matrix: {end_time - start_time:.4f} seconds")
        
        start_time = time.time()
        self.xp = F @ self.x
        self.Pp = F @ self.P @ F.T + Q
        end_time = time.time()
        print(f"Time taken to get dot products: {end_time - start_time:.4f} seconds")

    def update(self, z, z_err, psr_index):
        """Update the state and covariance with a new observation."""
        # Define the time-dependent H and R matrices for this timestep
        self.H = self.model.H_matrix(psr_index)
        self.R = self.model.R_matrix(z_err, psr_index)

        # Now run through the update algorithm
        y = z - self.H @ self.xp  # innovation
        S = self.H @ self.Pp @ self.H.T + self.R  # innovation covariance
        K = self.Pp @ self.H.T / S  # Kalman gain for scalar covariance
        self.x = self.xp + K * y  # updated state
        self.P = (np.eye(len(self.xp)) - K @ self.H) @ self.Pp  # updated covariance
        self.ll += self._log_likelihood(y, S)  # update the likelihood

    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        # Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
        self.model.set_global_parameters(θ)

        # Initialise x and P, the likelihood, and the index i
        self.x, self.P, self.ll, i = self.x0, self.P0, 0.0, int(0)


        # Do the first update step
        ##For the first update step, just assign the predicted values to be the states
        self.xp, self.Pp = self.x, self.P
        ##Update step
        start_time = time.time()
        self.update(self.data[i], self.data_errors[i], self.psr_indices[i])  # Updates x,P,and the likelihood_value
        end_time = time.time()
        print(f"Time taken to get first update: {end_time - start_time:.4f} seconds")

        # Replace the print with tqdm progress bar
        for i in tqdm(range(1, self.N_timesteps), desc="Processing timesteps"):
            # Set the delta t
            dt = self.t_diffs[i - 1]  

            # Predict step
            start_time = time.time()
            self.predict(dt)
            end_time = time.time()
            print(f"Time taken to get predict: {end_time - start_time:.4f} seconds")
            # Update step
        
            start_time = time.time()
            self.update(self.data[i], self.data_errors[i], self.psr_indices[i])  # Updates x,P,and the likelihood_value
            end_time = time.time()
            print(f"Time taken to get update: {end_time - start_time:.4f} seconds")

        return self.ll



class PartitionedKalmanFilter:
    """A class to implement the partitioned linear Kalman filter on scalar inputs.

    This implementation is similar to ScalarKalmanFilter but allows for custom
    predict step implementations to optimize matrix operations by exploiting
    the block structure of the matrices.

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
        return -0.5 * (np.log(2.0 * np.pi * cov) + (y * y) / cov)

    def predict(self, dt):
        """Predict the next state and covariance.
        
        This method should be implemented to optimize matrix operations
        by exploiting the block structure of the matrices.
        """
        
        # 1) Predict the mean
        a_pred = model.F_aa(dt) @ a_k
        x_pred = F_xa @ a_k + F_xx @ x_k

        # 2) Predict the covariance
        P_pred = F_aa @ P_aa @ F_aa.T + F_ax @ P_ax @ F_xa.T + F_ax @ P_xa @ F_xa.T + F_xx @ P_xx @ F_xx.T





        pass

    def update(self, z, z_err, psr_index):
        """Update the state and covariance with a new observation."""
        # Define the time-dependent H and R matrices for this timestep
        self.H = self.model.H_matrix(psr_index)
        self.R = self.model.R_matrix(z_err, psr_index)

        # Now run through the update algorithm
        y = z - self.H @ self.xp  # innovation
        S = self.H @ self.Pp @ self.H.T + self.R  # innovation covariance
        K = self.Pp @ self.H.T / S  # Kalman gain for scalar covariance
        self.x = self.xp + K * y  # updated state
        self.P = (np.eye(len(self.xp)) - K @ self.H) @ self.Pp  # updated covariance
        self.ll += self._log_likelihood(y, S)  # update the likelihood

    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        # Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
        self.model.set_global_parameters(θ)

        # Initialise x and P, the likelihood, and the index i
        self.x, self.P, self.ll, i = self.x0, self.P0, 0.0, int(0)

        # Do the first update step
        ##For the first update step, just assign the predicted values to be the states
        self.xp, self.Pp = self.x, self.P
        ##Update step
        self.update(self.data[i], self.data_errors[i], self.psr_indices[i])  # Updates x,P,and the likelihood_value

        # Replace the print with tqdm progress bar
        for i in tqdm(range(1, self.N_timesteps), desc="Processing timesteps"):
            # Set the delta t
            dt = self.t_diffs[i - 1]  

            # Predict step
            self.predict(dt)
            # Update step
            self.update(self.data[i], self.data_errors[i], self.psr_indices[i])  # Updates x,P,and the likelihood_value

        return self.ll
