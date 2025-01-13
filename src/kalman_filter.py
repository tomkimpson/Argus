"""Module which implements Kalman filter algorithm."""

import numpy as np 
import scipy



class ExtendedKalmanFilter:
    """A class to implement the extended (non-linear) Kalman filter.

    It takes four initialisation arguments:

        `Model`: class which defines all the Kalman machinery e.g. state transition models, covariance matrices etc. 

        `Observations`: 2D array which holds the noisy observations recorded at the detector

        `x0`: A 1D array which holds the initial guess of the initial states

        `P0`: The uncertainty in the guess of P0

    ...and a placeholder **kwargs, which is not currently used. 

    """

    def __init__(self,Model,Observations,x0,P0,**kwargs):
        """Initialize the class."""
        self.model        = Model
        self.observations = Observations


   
        assert self.observations.ndim == 2, f'This filter requires that input observations is a 2D array. The observations here have {self.observations.ndim} dimensions '
      
     
        self.n_measurement_states  = self.observations.shape[-1] #number of measurement states
        self.n_steps               = self.observations.shape[0]  #number of observations/timesteps
        self.n_states              = self.model.n_states


 
        self.x0 = x0 #Guess of the initial state 
        self.P0 = P0 #Guess of the initial state covariance

      



    """
    Given the innovation and innovation covariance, get the likelihood.
    """
    def _log_likelihood(self,y,cov):
        N = len(y)
        sign, log_det = np.linalg.slogdet(cov)
        ll = -0.5 * (log_det + np.dot(y.T, np.linalg.solve(cov, y))+ N*np.log(2 * np.pi))
        return ll


    """
    Predict step.
    """
    def _predict(self,x,P,parameters):
        f_function = self.model.f(x,parameters['g'])


        F_jacobian = self.model.F_jacobian(x,parameters['g'])
        Q          = self.model.Q_matrix(x,parameters['σp'])
    
        x_predict = f_function
        P_predict = F_jacobian@P@F_jacobian.T + Q

      
        return x_predict, P_predict




    """
    Update step
    """
    def _update(self,x, P, observation):

        #Evaluate the Jacobian of the measurement matrix
        H_jacobian = self.model.H_jacobian(x)

        #Now do the standard Kalman steps
        y_predicted = self.model.h(x)                       # The predicted y
        y           = observation - y_predicted             # The innovation/residual w.r.t actual data         
        S           = H_jacobian@P@H_jacobian.T + self.R    # Innovation covariance
        Sinv        = scipy.linalg.inv(S)                   # Innovation covariance inverse
        K           = P@H_jacobian.T@Sinv                   # Kalman gain
        xnew        = x + K@y                               # update x
        Pnew        = P -K@S@K.T                            # Update P 
        
        ##-------------------
        # The Pnew update equation is equation 5.27 in Sarkka  
        # Other equivalent options for computing Pnew are:
            # P = (I-KH)P
            # P = (I-KH)P(I-KH)' + KRK' (this one may have advantages for numerical stability)
        ##-------------------


        ll          = self._log_likelihood(y,S)          # and get the likelihood
        y_updated   = self.model.h(xnew)                 # and map xnew to measurement space


        return xnew, Pnew,ll,y_updated


    
    def run(self,parameters):
        """Run the Kalman filter and return a likelihood."""
        #Initialise x and P
        x = self.x0 
        P = self.P0

        #Initialise the likelihood
        self.ll = 0.0

        #Define any matrices which are constant in time
        self.R          = self.model.R_matrix(parameters['σm'])

     
        #Define arrays to store results
        self.state_predictions       = np.zeros((self.n_steps,self.n_states))
        self.measurement_predictions = np.zeros((self.n_steps,self.n_measurement_states))
        self.state_covariance        = np.zeros((self.n_steps,self.n_states,self.n_states))


        # #Do the first update step
        i = 0
        x,P,likelihood_value,y_predicted = self._update(x,P, self.observations[i,:])
        self.state_predictions[i,:] = x
        self.state_covariance[i,:,:] = P
        self.measurement_predictions[i,:]  = y_predicted
        self.ll +=likelihood_value

 
     
        for i in np.arange(1,self.n_steps):

            #Predict step
            x_predict, P_predict             = self._predict(x,P,parameters)                                        # The predict step
            
            #Update step
            x,P,likelihood_value,y_predicted = self._update(x_predict,P_predict, self.observations[i,:]) # The update step
            
            #Update the running sum of the likelihood and save the state and measurement predictions
            self.ll +=likelihood_value
            self.state_predictions[i,:] = x
            self.state_covariance[i,:,:] = P
            self.measurement_predictions[i,:]  = y_predicted
       

