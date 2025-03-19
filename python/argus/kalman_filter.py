"""Module which implements Kalman filter algorithm."""

import numpy as np
from tqdm import tqdm
from argus.jmath import get_Pp, get_xp, get_P_blocks, get_Pp_blocks
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
        F_list = self.model.F_matrix(dt) # (F_gw, F_spin)
        Q_list = self.model.Q_matrix(dt) # (Q_gw, Q_spin, Q_timing)
        self.xp = get_xp(F_list, self.x, 72, 72)
        P_list = get_P_blocks(self.P, 72, 72)
        self.Pp = get_Pp_blocks(F_list, P_list, Q_list)
        # breakpoint()

        # self.xp = F @ self.x
        # # self.Pp = F @ self.P @ F.T + Q
        # self.Pp = get_Pp(F, self.P, Q)
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




    def kalman_update_scalar_symmetric(
        # --- State blocks ---
        x_gw:     np.ndarray,  # shape (2N,)
        x_spin:   np.ndarray,  # shape (2N,)
        x_eps:    np.ndarray,  # shape (sum(M^{(n)}),)

        # --- Covariance blocks (only storing one side of cross terms) ---
        P_gw:        np.ndarray,  # (2N, 2N)
        P_gw_spin:   np.ndarray,  # (2N, 2N) = cross-block for (GW,spin)
        P_gw_eps:    np.ndarray,  # (2N, sum(M))
        P_spin:      np.ndarray,  # (2N, 2N)
        P_spin_eps:  np.ndarray,  # (2N, sum(M))
        P_eps:       np.ndarray,  # (sum(M), sum(M))

        # --- Measurement noise, scalar measurement ---
        R: float,  # measurement noise variance
        y: float   # scalar measurement
    ):
        """
        Perform one Kalman update for a single scalar measurement y

        Returns
        -------
        Updated (x_gw, x_spin, x_eps, P_gw, P_gw_spin, P_gw_eps, P_spin, P_spin_eps, P_eps).
        """

        # 1) Innovation (residual)
        #    nu = y - [h_gw @ x_gw + h_spin @ x_spin + h_eps @ x_eps]
        nu = y - (h_gw @ x_gw + h_spin @ x_spin + h_eps @ x_eps)

        # 2) Vector "u" = P * H^T, in block form.
        #    Each u_* is the partial result for that block of the state.
        u_gw   = P_gw @ h_gw + P_gw_spin @ h_spin + P_gw_eps @ h_eps
        u_spin = P_gw_spin.T @ h_gw + P_spin @ h_spin + P_spin_eps @ h_eps
        # Note: P_gw_spin is (2N,2N), so P_gw_spin.T is (2N,2N)
        # We do not store P_spin_gw, so we use the transpose here.
        u_eps  = P_gw_eps.T @ h_gw + P_spin_eps.T @ h_spin + P_eps @ h_eps
        # Similarly, P_gw_eps is (2N, sum(M)), so P_gw_eps.T is (sum(M), 2N), etc.

        # 3) Innovation variance: S = H * P * H^T + R
        S = (h_gw @ u_gw) + (h_spin @ u_spin) + (h_eps @ u_eps) + R

        # 4) Kalman gain scale
        alpha = 1.0 / S

        # 5) Updated state
        x_gw_up   = x_gw   + alpha * u_gw   * nu
        x_spin_up = x_spin + alpha * u_spin * nu
        x_eps_up  = x_eps  + alpha * u_eps  * nu

        # 6) Rank-1 covariance update
        #    P <- P - alpha * u * u^T, done in block form.
        #    We'll do each "kept" block, then set its symmetric counterpart.

        # 6a) GW-GW block
        P_gw_up = P_gw - alpha * np.outer(u_gw, u_gw)

        # 6b) GW-Spin block (we store only P_gw_spin, mirror is P_gw_spin_up.T)
        P_gw_spin_up = P_gw_spin - alpha * np.outer(u_gw, u_spin)

        # 6c) Spin-Spin block
        P_spin_up = P_spin - alpha * np.outer(u_spin, u_spin)

        # 6d) GW-Eps block
        P_gw_eps_up = P_gw_eps - alpha * np.outer(u_gw, u_eps)

        # 6e) Spin-Eps block
        P_spin_eps_up = P_spin_eps - alpha * np.outer(u_spin, u_eps)

        # 6f) Eps-Eps block
        P_eps_up = P_eps - alpha * np.outer(u_eps, u_eps)


        return (
            x_gw_up, x_spin_up, x_eps_up,
            P_gw_up, P_gw_spin_up, P_gw_eps_up,
            P_spin_up, P_spin_eps_up, P_eps_up
        )

        
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
