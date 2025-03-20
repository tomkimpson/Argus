"""Module which implements Kalman filter algorithm."""

import numpy as np
from tqdm import tqdm
from line_profiler import profile
import jax.numpy as jnp
from jax import jit

def get_ith_pair(x,i):
    """
    Return the ith entries of x.
    """
    idx = 2 * i
    return x[idx], x[idx + 1]

def get_ith_vector(x, M_cumsum, i):
    return x[M_cumsum[i]:M_cumsum[i + 1]]


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


        #new stuff
        self.pulsar_design_matrices = self.model.pulsar_design_matrices
        self.design_matrix_counter = np.zeros(self.model.Npsr)


        #Initialise the state vectors
        self.x_gw0,self.x_spin0,self.x_eps0 =  x0
        self.P_gw0,self.P_spin0,self.P_eps0,self.P_gw_spin0,self.P_gw_eps0,self.P_spin_eps0 =  P0



        assert np.isscalar(self.data[0])

    def _log_likelihood(self, y, cov):
        """Given the innovation and innovation covariance, get the likelihood."""
        log_likelihood = -0.5 * (np.log(2.0 * np.pi * cov) + (y * y) / cov)
        return log_likelihood


    # @profile
    # def predict(self, dt):
    #     """Predict the next state and covariance."""
    #     F_list = self.model.F_matrix(dt) # (F_gw, F_spin)
    #     Q_list = self.model.Q_matrix(dt) # (Q_gw, Q_spin, Q_timing)
    #     self.xp = get_xp(F_list, self.x, 72, 72)
    #     P_list = get_P_blocks(self.P, 72, 72)
    #     self.Pp = get_Pp_blocks(F_list, P_list, Q_list)

    def predict(self, dt,
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
    ):
    
        

        #Define the F matrices for each block
        F_gw, F_spin,F_eps = self.model.F_matrix(dt)

        #Predict the next state
        x_gw_predict = F_gw@x_gw
        x_spin_predict =  F_spin@x_spin
        x_eps_predict = F_eps@x_eps

        #Define the Q matrices for each block
        Q_gw, Q_spin, Q_eps = self.model.Q_matrix(dt)


        #Predict the next covariance

        ## auto covariance terms
        P_gw_predict     = F_gw@P_gw@F_gw.T + Q_gw
        P_spin_predict   = F_spin@P_spin@F_spin.T + Q_spin
        P_eps_predict    = F_eps@P_eps@F_eps.T + Q_eps

        ## cross covariance terms
        P_gw_spin_predict  = F_gw@P_gw_spin@F_spin.T 
        P_gw_eps_predict   = F_gw@P_gw_eps@F_eps.T 
        P_spin_eps_predict = F_spin@P_spin_eps@F_eps.T 


        return (
            x_gw_predict,
            x_spin_predict,
            x_eps_predict,
            P_gw_predict,
            P_gw_spin_predict,
            P_gw_eps_predict,
            P_spin_predict,
            P_spin_eps_predict,
            P_eps_predict
        )






    def update(
        self,
        psr_index,

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

        #Define the extra covariance terms. We can probably drop this and just use the symmetric covariances
        P_spin_gw  = P_gw_spin.T
        P_eps_gw   = P_gw_eps.T
        P_eps_spin = P_spin_eps.T

        #Define some relevant parameters for this pulsar and timestep


        row_idx = int(self.design_matrix_counter[psr_index])
        f0 = self.model.f0[psr_index]
        M = self.pulsar_design_matrices[psr_index][row_idx,:] #self.pulsar_design_matrices is a list of design matrices for each pulsar, length Npsr. And individual design mastrix has shape (Ntimesteps,Nparameters)

        # 1) Innovation (residual)

        r  = get_ith_pair(x_gw, psr_index)[0] #scalar
        δφ = get_ith_pair(x_spin, psr_index)[0] #scalar
        δε = get_ith_vector(x_eps, self.model.M_cumsum, psr_index) #vector


        nu = y - (-r + δφ /f0 + M@δε) #write out the measurement equation explicitly
        self.design_matrix_counter[psr_index] += 1 #increment the design matrix counter for this pulsar


        # 2) The vector "u_gw" = P_gw @ h_gw^T, but h_gw^T has only one nonzero at col_r_n
        #    with scale val_r_n. So we pick that column from P_gw:

        u_gw   = -P_gw[:,psr_index]      + P_gw_spin[:,psr_index]/f0  + P_gw_eps[:, self.model.M_cumsum[psr_index] : self.model.M_cumsum[psr_index+1]]@M
        u_spin = -P_spin_gw[:,psr_index] + P_spin[:,psr_index]/f0     + P_spin_eps[:, self.model.M_cumsum[psr_index] : self.model.M_cumsum[psr_index+1]]@M
        u_eps  = -P_eps_gw[:,psr_index]  + P_eps_spin[:,psr_index]/f0 + P_eps[:, self.model.M_cumsum[psr_index] : self.model.M_cumsum[psr_index+1]]@M


    

        # 3) Innovation variance: S = H * P * H^T + R
        #S = (h_gw @ u_gw) + (h_spin @ u_spin) + (h_eps @ u_eps) + R
        # Note: is slicing like this any more efficient that creating the h vectors?
        u_gw_value   = get_ith_pair(u_gw, psr_index)[0] # get (r,a) pair then select the r value. Equivalent to h_gw @ u_gw priduct as h_gw is all zeros apart from -1 factor
        u_spin_value = get_ith_pair(u_spin, psr_index)[0]
        u_eps_value  = get_ith_vector(u_eps, self.model.M_cumsum, psr_index)
        S = (-1 * u_gw_value) + (1.0/f0 * u_spin_value) + (M @ u_eps_value) + R
    
       
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
    @jit
    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        # Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
        self.model.set_global_parameters(θ)

        # Initialise x and P, the likelihood, and the index i
 
        #Initialise the state vectors
        # Handles the states in an explicit block format
        # Covariances are handled similarly


        #Initialise the likelihood
        self.ll = 0.0

        #Initialise the index
        i = 0
        # Update step
        #Need to correct self handiling, but leaving returns explicit for now
        (x_gw, x_spin, x_eps,
         P_gw, P_gw_spin, P_gw_eps,
         P_spin, P_spin_eps, P_eps) = self.update(
            psr_index=int(self.psr_indices[i]),
            x_gw=self.x_gw0,
            x_spin=self.x_spin0,
            x_eps=self.x_eps0,
            P_gw=self.P_gw0,
            P_gw_spin=self.P_gw_spin0,
            P_gw_eps=self.P_gw_eps0,
            P_spin=self.P_spin0,
            P_spin_eps=self.P_spin_eps0,
            P_eps=self.P_eps0,
            R=self.data_errors[i],
            y=self.data[i]
        )



        # Iterate over the data.
        # tqdm progress bar just for testing. To be removed ultimately
        for i in tqdm(range(1, self.N_timesteps), desc="Processing timesteps"):
            # Set the delta t
            dt = self.t_diffs[i - 1]

            # Predict step
            (x_gw_predict, x_spin_predict, x_eps_predict,
                P_gw_predict, P_gw_spin_predict, P_gw_eps_predict,
                P_spin_predict, P_spin_eps_predict, P_eps_predict) = self.predict(dt,
                x_gw      = x_gw,
                x_spin    = x_spin,
                x_eps     = x_eps,
                P_gw      = P_gw,
                P_gw_spin = P_gw_spin,
                P_gw_eps  = P_gw_eps,
                P_spin    = P_spin,
                P_spin_eps= P_spin_eps,
                P_eps     = P_eps
            )

            # Update step
            (x_gw_up, x_spin_up, x_eps_up,
                P_gw_up, P_gw_spin_up, P_gw_eps_up,
                P_spin_up, P_spin_eps_up, P_eps_up) = self.update(
                psr_index=int(self.psr_indices[i]),
                x_gw        = x_gw_predict,
                x_spin      = x_spin_predict,
                x_eps       = x_eps_predict,
                P_gw        = P_gw_predict,
                P_gw_spin   = P_gw_spin_predict,
                P_gw_eps    = P_gw_eps_predict,
                P_spin      = P_spin_predict,
                P_spin_eps  = P_spin_eps_predict,
                P_eps       = P_eps_predict,
                R          = self.data_errors[i],
                y          = self.data[i]
            )




        return self.ll
