"""Module which implements Kalman filter algorithm."""

import numpy as np 


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
        self.model        = model
        self.observations = observations 
        self.x0           = x0
        self.P0           = P0


        #Extract the observations into separate arrays
        self.toa         = self.observations[:,0]
        self.data        = self.observations[:,1]
        self.data_errors = self.observations[:,2]
        self.psr_indices = self.observations[:,3].astype(int)
        self.N_timesteps = len(self.observations)
        self.t_diffs     = np.diff(self.toa)

        assert  np.isscalar(self.data[0])



    def _log_likelihood(self,y,cov):
        """Given the innovation and innovation covariance, get the likelihood."""
        return -0.5 * (np.log(2.0 * np.pi * cov) +(y * y) / cov) 


    def predict(self,dt):
        """Predict the next state and covariance."""
        F = self.model.F_matrix(dt)
        Q = self.model.Q_matrix(dt)

    
        self.xp = F @ self.x 
        self.Pp = F @ self.P @ F.T + Q

    def update(self, z,z_err,psr_index):
        """Update the state and covariance with a new observation."""
        #Define the time-dependent H and R matrices for this timestep
        self.H = self.model.H_matrix(psr_index)
        self.R = self.model.R_matrix(z_err,psr_index)

        #Now run through the update algorithm
        y      = z - self.H @ self.xp                                # innovation
        S      = self.H @ self.Pp @ self.H.T + self.R                # innovation covariance
        K      = self.Pp @ self.H.T / S                              # Kalman gain for scalar covariance
        self.x = self.xp + K * y                                     # updated state
        self.P = (np.eye(len(self.xp)) - K @ self.H) @ self.Pp       # updated covariance
        self.ll += self._log_likelihood(y,S)                         # update the likelihood


    def get_likelihood(self,θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        #Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
        self.model.set_global_parameters(θ) 


        #Initialise x and P, the likelihood, and the index i
        self.x,self.P,self.ll,i = self.x0,self.P0,0.0,int(0)  
 
          
        #Do the first update step
        ##For the first update step, just assign the predicted values to be the states
        self.xp,self.Pp = self.x, self.P 
        ##Update step
        self.update(self.data[i],self.data_errors[i],self.psr_indices[i]) # Updates x,P,and the likelihood_value 

        print(i)
        for i in range(1,self.N_timesteps): #indexing starts at 1 as we have already done i=0
            #print(i)
            #Set the delta t
            dt = self.t_diffs[i-1] #For example, when i=1, we use the 0th element of t_diffs for the predict step
            
            #Predict step
            self.predict(dt)
            #Update step
            self.update(self.data[i],self.data_errors[i],self.psr_indices[i]) # Updates x,P,and the likelihood_value 

            


class JAXKalmanFilter:
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
        self.model        = model
        self.observations = observations 
        self.x0           = x0
        self.P0           = P0


        #Extract the observations into separate arrays
        self.toa         = self.observations[:,0]
        self.data        = self.observations[:,1]
        self.data_errors = self.observations[:,2]
        self.psr_indices = self.observations[:,3].astype(int)
        self.N_timesteps = len(self.observations)
        self.t_diffs     = np.diff(self.toa)

        assert  np.isscalar(self.data[0])



    # def _log_likelihood(self,y,cov):
    #     """Given the innovation and innovation covariance, get the likelihood."""
    #     return -0.5 * (np.log(2.0 * np.pi * cov) +(y * y) / cov) 




    # def predict_a(self,a, P_aa, dt):

    #     """
    #     Predict the global a-vector over a time-step dt.
    #     Returns a_pred, P_aa_pred.
    #     """
    #     # Example: an OU with damping gamma_a => F^a = e^{-gamma_a * dt} * I
    #     # Q^aa = integrated noise covariance
    #     F_a = self.model.build_F_a(dt)    # shape (N,N)
    #     Q_aa = self.model.build_Q_aa(dt)  # shape (N,N)

    #     a_pred = F_a @ a
    #     P_aa_pred = F_a @ P_aa @ F_a.T + Q_aa

    #     return a_pred, P_aa_pred




# def predict_local(n,
#                   x_n, P_xx_n, P_xa_n,
#                   a_pred, P_aa_pred,
#                   dt):
#     """
#     Predict pulsar n's local state x^(n), plus blocks P_{xx}^{(n)}, P_{xa}^{(n)},
#     given that 'a_pred' and 'P_aa_pred' are the already-predicted global blocks.
#     """
#     # 1) Build local transition F_n^x, local noise Q_n^x, coupling G_n
#     F_nx = build_F_nx(dt, n)    # shape (d_n, d_n)
#     Q_nx = build_Q_nx(dt, n)    # shape (d_n, d_n)
#     G_n  = build_G_n(dt, n)     # shape (d_n, N)

#     # 2) Mean
#     x_n_pred = F_nx @ x_n + G_n @ a_pred  # we use the *pred* a

#     # 3) Cov: 
#     #   P_xx_n_pred = F_nx P_xx_n F_nx^T + etc. (some cross terms omitted for brevity)
#     P_xx_n_pred = F_nx @ P_xx_n @ F_nx.T + Q_nx
#     #   plus G_n P_aa_pred G_n^T if needed, and cross terms, etc.
#     P_xx_n_pred += G_n @ P_aa_pred @ G_n.T

#     # 4) Cross-cov P_{xa}^{(n)}
#     #   typically P_{xa}^{(n)}_pred = F_nx * P_{xa}^{(n)} * F_a^T + G_n * P_aa * F_a^T
#     #   but if we've already updated a->a_pred, be consistent with the new P_aa_pred etc.
#     #   This depends on whether you're doing the block-lift carefully. 
#     # For a simple approach, we can do:
#     P_xa_n_pred = F_nx @ P_xa_n

#     return x_n_pred, P_xx_n_pred, P_xa_n_pred




# def predict_partitioned(a, P_aa,
#                         x_list, P_xx_list, P_xa_list,
#                         dt):
#     """
#     Partitioned predict step:
#       1) Predict global a, P_aa
#       2) For each pulsar n, predict x^(n), P_{xx}^{(n)}, P_{xa}^{(n)} 
#     """
#     # (1) global
#     a_new, P_aa_new = predict_a(a, P_aa, dt)

#     # (2) local
#     x_list_new = []
#     P_xx_list_new = []
#     P_xa_list_new = []


#     #loop over all pulsars
#     #independent, amenable to parallelization
#     for n in range(len(x_list)): 
#         x_n = x_list[n]
#         P_xx_n = P_xx_list[n]
#         P_xa_n = P_xa_list[n]  # shape (d_n, N)

#         x_n_pred, P_xx_n_pred, P_xa_n_pred = predict_local(
#             n,
#             x_n, P_xx_n, P_xa_n,
#             a_new, P_aa_new,
#             dt
#         )

#         x_list_new.append(x_n_pred)
#         P_xx_list_new.append(P_xx_n_pred)
#         P_xa_list_new.append(P_xa_n_pred)

#     return a_new, P_aa_new, x_list_new, P_xx_list_new, P_xa_list_new




# # Example: build H_x for pulsar n’s local state
# # Indices: 0=phi, 1=r, 2..(1+M)=eps
# def build_measurement_matrix_xn(f0, M_vec, local_dim_n):
#     # local_dim_n = 2 + M   # (phi, r, M eps)
#     H_x = np.zeros((1, local_dim_n))  # shape (1, 2+M)
#     # phi coefficient = 1/f0
#     H_x[0, 0] = 1.0 / f0
#     # r coefficient = -1
#     H_x[0, 1] = -1.0
#     # eps coefficients = M_vec (size M)
#     # Place them in columns 2..(1+M)
#     for i, val in enumerate(M_vec):
#         H_x[0, 2 + i] = val
#     return H_x


# def update_single_pulsar(a, P_aa,
#                          x_n, P_xx_n, P_xa_n,
#                          f0, M_vec,
#                          y_obs, R_obs):
#     """
#     Perform a single-scalar measurement update:
#       delta t = phi/f0 + M * eps - r
#     ignoring direct a in the measurement equation,
#     but a is correlated with r => cross-cov.
#     """

#     # 1) Build the local measurement matrix H_x
#     #    x_n = (phi, r, eps1, ..., epsM)
#     ld_n = x_n.shape[0]  # should be 2 + M
#     H_x = build_measurement_matrix_xn(f0, M_vec, ld_n)  # shape (1, ld_n)

#     # 2) Predicted measurement
#     #    y_pred = H_x @ x_n  (assuming no direct a in eqn)
#     #    if a is directly in the measurement, we would build H_a, etc.
#     y_pred = (H_x @ x_n.reshape(-1,1))[0,0]
#     resid = y_obs - y_pred

#     # 3) Innovation variance:
#     # We have a big 2-block: {a, x_n}.
#     # But if "H_a=0", the measurement sees only x_n. Then "S" depends on P_xx_n...
#     # For a more general case, let's define H_a=0 if we do not measure a directly.
#     H_a = np.zeros((1, a.shape[0]))  # (1, N)

#     # S = H_a P_aa H_a^T + H_a P_a^x H_x^T
#     #   + H_x P_x^a H_a^T + H_x P_xx H_x^T + R_obs
#     #   We'll do it piecewise.
#     S_val = (H_a @ P_aa @ H_a.T +
#              H_a @ P_xa_n.T @ H_x.T +  # P_a^x => P_xa_n^T
#              H_x @ P_xa_n @ H_a.T +
#              H_x @ P_xx_n @ H_x.T +
#              R_obs)
#     S_scalar = S_val[0,0]
#     S_inv = 1.0 / S_scalar

#     # 4) Kalman gain for a, x_n
#     # K_a = P_aa H_a^T + P_a^x H_x^T
#     K_a = (P_aa @ H_a.T) + (P_xa_n.T @ H_x.T)   # shape(N,1)
#     K_a *= S_inv

#     # K_x = P_xa_n H_a^T + P_xx_n H_x^T
#     K_x = (P_xa_n @ H_a.T) + (P_xx_n @ H_x.T)   # shape(ld_n,1)
#     K_x *= S_inv

#     # 5) Posterior means
#     a_post = a + K_a[:,0] * resid
#     x_n_post = x_n + K_x[:,0] * resid

#     # 6) Posterior cov (rank-1 update on the 2-block)
#     #   P_aa_post = P_aa - K_a * (H_a P_aa + H_x P_xa^T)
#     # etc.  We do partial expansions:
#     # We'll define "H_a P_aa" as (1,N), etc.

#     # For brevity:
#     HaPaa = H_a @ P_aa
#     HaPxa = H_a @ P_xa_n.T   # shape(1, ld_n)
#     HxPxa = H_x @ P_xa_n     # shape(1, N)
#     HxPxx = H_x @ P_xx_n     # shape(1, ld_n)

#     # delta P_aa = K_a outer [HaPaa + HxPxa]
#     # do a small helper to get the outer product
#     dP_aa = np.outer(K_a, (HaPaa + HxPxa))
#     P_aa_post = P_aa - dP_aa

#     # P_xa_n_post
#     #   P_xa_n -> P_xa_n - [K_x outer (HaPaa + HxPxa)]
#     dP_xa = np.outer(K_x, (HaPaa + HxPxa))
#     P_xa_n_post = P_xa_n - dP_xa

#     # P_xx_n_post
#     #   P_xx_n -> P_xx_n - K_x outer (HaPxa + HxPxx)
#     dP_xx = np.outer(K_x, (HaPxa + HxPxx))
#     P_xx_n_post = P_xx_n - dP_xx

#     return a_post, P_aa_post, x_n_post, P_xx_n_post, P_xa_n_post


    # def get_likelihood(self,θ):
    #     """Run the Kalman filter algorithm over all observations and return a log likelihood."""
    #     #Define all the free parameters for the model. Note this exludes dt, which is not a parameter we need to infer.
    #     self.model.set_global_parameters(θ) 


    #     #Initialise x and P, the likelihood, and the index i
    #     self.x,self.P,self.ll,i = self.x0,self.P0,0.0,int(0)  
 
          
    #     #Do the first update step
    #     ##For the first update step, just assign the predicted values to be the states
    #     self.xp,self.Pp = self.x, self.P 
    #     ##Update step
    #     self.update(self.data[i],self.data_errors[i],self.psr_indices[i]) # Updates x,P,and the likelihood_value 

    #     print(i)
    #     for i in range(1,3): #indexing starts at 1 as we have already done i=0
    #         #print(i)
    #         #Set the delta t
    #         dt = self.t_diffs[i-1] #For example, when i=1, we use the 0th element of t_diffs for the predict step
            
    #         #Predict step
    #         self.predict(dt)
    #         #Update step
    #         self.update(self.data[i],self.data_errors[i],self.psr_indices[i]) # Updates x,P,and the likelihood_value 

            



# class ExtendedKalmanFilter:
#     """A class to implement the extended (non-linear) Kalman filter.

#     It takes four initialisation arguments:

#         `Model`: class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc. 

#         `Observations`: 2D array which holds the noisy observations recorded at the detector

#         `x0`: A 1D array which holds the initial guess of the initial states

#         `P0`: The uncertainty in the guess of P0

#     ...and a placeholder **kwargs, which is not currently used. 

#     """

#     def __init__(self,Model,Observations,x0,P0,**kwargs):
#         """Initialize the class."""
#         self.model        = Model
#         self.observations = Observations


   
#         assert self.observations.ndim == 2, f'This filter requires that input observations is a 2D array. The observations here have {self.observations.ndim} dimensions '
      
     
#         self.n_measurement_states  = self.observations.shape[-1] #number of measurement states
#         self.n_steps               = self.observations.shape[0]  #number of observations/timesteps
#         self.n_states              = self.model.n_states


 
#         self.x0 = x0 #Guess of the initial state 
#         self.P0 = P0 #Guess of the initial state covariance

      



#     """
#     Given the innovation and innovation covariance, get the likelihood.
#     """
#     def _log_likelihood(self,y,cov):
#         N = len(y)
#         sign, log_det = np.linalg.slogdet(cov)
#         ll = -0.5 * (log_det + np.dot(y.T, np.linalg.solve(cov, y))+ N*np.log(2 * np.pi))
#         return ll


#     """
#     Predict step.
#     """
#     def _predict(self,x,P,parameters):
#         f_function = self.model.f(x,parameters['g'])


#         F_jacobian = self.model.F_jacobian(x,parameters['g'])
#         Q          = self.model.Q_matrix(x,parameters['σp'])
    
#         x_predict = f_function
#         P_predict = F_jacobian@P@F_jacobian.T + Q

      
#         return x_predict, P_predict




#     """
#     Update step
#     """
#     def _update(self,x, P, observation):

#         #Evaluate the Jacobian of the measurement matrix
#         H_jacobian = self.model.H_jacobian(x)

#         #Now do the standard Kalman steps
#         y_predicted = self.model.h(x)                       # The predicted y
#         y           = observation - y_predicted             # The innovation/residual w.r.t actual data         
#         S           = H_jacobian@P@H_jacobian.T + self.R    # Innovation covariance
#         Sinv        = scipy.linalg.inv(S)                   # Innovation covariance inverse
#         K           = P@H_jacobian.T@Sinv                   # Kalman gain
#         xnew        = x + K@y                               # update x
#         Pnew        = P -K@S@K.T                            # Update P 
        
#         ##-------------------
#         # The Pnew update equation is equation 5.27 in Sarkka  
#         # Other equivalent options for computing Pnew are:
#             # P = (I-KH)P
#             # P = (I-KH)P(I-KH)' + KRK' (this one may have advantages for numerical stability)
#         ##-------------------


#         ll          = self._log_likelihood(y,S)          # and get the likelihood
#         y_updated   = self.model.h(xnew)                 # and map xnew to measurement space


#         return xnew, Pnew,ll,y_updated


    
#     def run(self,parameters):
#         """Run the Kalman filter and return a likelihood."""
#         #Initialise x and P
#         x = self.x0 
#         P = self.P0

#         #Initialise the likelihood
#         self.ll = 0.0

#         #Define any matrices which are constant in time
#         self.R          = self.model.R_matrix(parameters['σm'])

     
#         #Define arrays to store results
#         self.state_predictions       = np.zeros((self.n_steps,self.n_states))
#         self.measurement_predictions = np.zeros((self.n_steps,self.n_measurement_states))
#         self.state_covariance        = np.zeros((self.n_steps,self.n_states,self.n_states))


#         # #Do the first update step
#         i = 0
#         x,P,likelihood_value,y_predicted = self._update(x,P, self.observations[i,:])
#         self.state_predictions[i,:] = x
#         self.state_covariance[i,:,:] = P
#         self.measurement_predictions[i,:]  = y_predicted
#         self.ll +=likelihood_value

 
     
#         for i in np.arange(1,self.n_steps):

#             #Predict step
#             x_predict, P_predict             = self._predict(x,P,parameters)                                        # The predict step
            
#             #Update step
#             x,P,likelihood_value,y_predicted = self._update(x_predict,P_predict, self.observations[i,:]) # The update step
            
#             #Update the running sum of the likelihood and save the state and measurement predictions
#             self.ll +=likelihood_value
#             self.state_predictions[i,:] = x
#             self.state_covariance[i,:,:] = P
#             self.measurement_predictions[i,:]  = y_predicted
       

