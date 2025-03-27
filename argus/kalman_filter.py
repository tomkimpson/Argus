class KalmanFilter:
    """A Kalman Filter implementation using NumPy for state estimation and likelihood computation."""
    
    def __init__(self, model, observations, x0, P0):
        """Initialize the Kalman Filter.
        
        Args:
            model: The state space model object containing transition and observation matrices
            observations: Array of observations
            x0: Initial state vector
            P0: Initial state covariance matrix
        """
        self.model = model
        self.observations = observations
        self.x = x0.copy()  # Current state estimate
        self.P = P0.copy()  # Current state covariance
        
    def predict(self, params):
        """Prediction step of the Kalman filter.
        
        Args:
            params: Parameter struct containing model parameters
            
        Returns:
            x_pred: Predicted state
            P_pred: Predicted state covariance
        """
        # Get transition matrix and process noise
        F = self.model.get_transition_matrix(params)
        Q = self.model.get_process_noise_matrix(params)
        
        # Predict state and covariance
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q
        
        return x_pred, P_pred
    
    def update(self, y, params, x_pred, P_pred):
        """Update step of the Kalman filter.
        
        Args:
            y: Current observation
            params: Parameter struct containing model parameters
            x_pred: Predicted state from predict step
            P_pred: Predicted covariance from predict step
            
        Returns:
            log_likelihood: Log-likelihood contribution of this observation
            x_new: Updated state estimate
            P_new: Updated state covariance
        """
        # Get observation matrix and noise
        H = self.model.get_observation_matrix(params)
        R = self.model.get_observation_noise_matrix(params)
        
        # Innovation and its covariance
        v = y - H @ x_pred
        S = H @ P_pred @ H.T + R
        
        # Ensure S is scalar since we're dealing with scalar observations
        S = np.atleast_1d(S)[0]
        
        # Kalman gain
        K = P_pred @ H.T / S
        
        # Update state and covariance
        x_new = x_pred + K * v
        P_new = P_pred - np.outer(K, H @ P_pred)
        
        # Compute log-likelihood contribution
        log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(S) + (v * v) / S)
        
        return log_likelihood, x_new, P_new
    
    def get_likelihood(self, params):
        """Compute the log-likelihood of the observations given the parameters.
        
        Args:
            params: Parameter struct containing model parameters
            
        Returns:
            float: Total log-likelihood
        """
        # Initialize
        log_likelihood = 0.0
        self.x = np.zeros_like(self.x)  # Reset state
        self.P = np.eye(len(self.x)) * 1e-12  # Reset covariance
        
        # Run filter
        for y in self.observations:
            # Predict
            x_pred, P_pred = self.predict(params)
            
            # Update
            ll, self.x, self.P = self.update(y, params, x_pred, P_pred)
            log_likelihood += ll
            
        return log_likelihood 