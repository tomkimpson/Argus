import os
import glob
from argus import data_loader, models, kalman_filter
import numpy as np
import pandas as pd
import cProfile
import pstats
import time

def hellings_downs(θ):
    """Compute the Hellings–Downs function for an angle θ (in radians).
    
    Parameters
    ----------
    θ : np.ndarray or float
        Angular separation between pulsars in radians
        
    Returns
    -------
    np.ndarray or float
        Hellings-Downs correlation values
    """
    # Handle the autocorrelation case first
    if isinstance(θ, np.ndarray):
        mask = np.isclose(θ, 0.0)
        x = np.zeros_like(θ)
        # Only compute (1-cos(θ))/2 for non-zero angles
        x[~mask] = (1 - np.cos(θ[~mask])) / 2.0
        
        result = np.zeros_like(θ)
        result[mask] = 1.0
        # Only compute HD function for non-zero angles
        result[~mask] = (3 / 2) * x[~mask] * np.log(x[~mask]) - x[~mask] / 4 + 0.5
        
        return result
    else:
        # Handle scalar input
        if np.isclose(θ, 0.0):
            return 1.0
        x = (1 - np.cos(θ)) / 2.0
        return (3 / 2) * x * np.log(x) - x / 4 + 0.5

def test_filter_run():
    """Test the KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""
    # Generate some data
    # Load some data to test on
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    directory = os.path.join(
        script_dir,
        "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/",
    )

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))
    assert len(par_files) == len(
        tim_files
    ), "Mismatch between .par and .tim file counts."

    # Get the data
    pulsar_residuals, pulsar_metadata = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            par_files[0:2], tim_files[0:2]
        )
    )

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    # Post-process the residuals
    processed_pulsar_residuals = (
        data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)
    )

    # Initialize the GW background model with HD correlation
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    KF = kalman_filter.ScalarKalmanFilter(
        model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0
    )

    # # Set global parameters.
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    KF.get_likelihood(params)


def test_filter_run_preprocessed():
    """Test the KalmanFilter class using preprocessed data from .npy and .parquet files."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    print(
        "Total length of the data is ", len(processed_pulsar_residuals)
    )
    print(
        "Total number of pulsars is ", len(pulsar_metadata)
    )

    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    print("Initialize the Kalman filter")
    KF = kalman_filter.ScalarKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals,
        x0=x0, 
        P0=P0
    )

    print("Set global params")
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    print("Run filter")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

def test_filter_single_timestep():
    """Test the KalmanFilter class using preprocessed data for a single timestep."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Take just the first timestep
    single_timestep_residuals = processed_pulsar_residuals[0:1]
    
    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    print("Initialize the Kalman filter")
    KF = kalman_filter.ScalarKalmanFilter(
        model=model, 
        observations=single_timestep_residuals,
        x0=x0, 
        P0=P0
    )

    print("Set global params")
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    print("Run single timestep")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood for single timestep: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

def test_filter_single_timestep_restructured():
    """Test the KalmanFilter class using preprocessed data for a single timestep with restructured model."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Take just the first timestep
    single_timestep_residuals = processed_pulsar_residuals[0:2]
    
    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    print("Initializing the model")
    model = models.SGWB_restructured(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    print("Initialize the Kalman filter")
    KF = kalman_filter.ScalarKalmanFilter(
        model=model, 
        observations=single_timestep_residuals,
        x0=x0, 
        P0=P0
    )

    print("Set global params")
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    print("Run single timestep")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood for single timestep: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

def test_filter_run_preprocessed_restructured():
    """Test the KalmanFilter class using preprocessed data with restructured model."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    print(
        "Total length of the data is ", len(processed_pulsar_residuals)
    )
    print(
        "Total number of pulsars is ", len(pulsar_metadata)
    )

    print("Initializing the model")
    model = models.SGWB_restructured(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    print("Initialize the Kalman filter")
    KF = kalman_filter.ScalarKalmanFilter(
        model=model, 
        observations=processed_pulsar_residuals,
        x0=x0, 
        P0=P0
    )

    print("Set global params")
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    print("Run filter")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

def test_compare_model_performance():
    """Compare performance between original and restructured models."""
    import time
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    # Initialize both models
    model_original = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix)
    model_restructured = models.SGWB_restructured(pulsar_metadata, hd_correlation_matrix)
    
    # Set parameters
    params = {
        "γa": 1e-1,
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }
    
    model_original.set_global_parameters(params)
    model_restructured.set_global_parameters(params)
    
    # Test F_matrix performance
    dt = 1.0
    n_iterations = 20
    
    print("\nTesting F_matrix performance:")
    start = time.time()
    for _ in range(n_iterations):
        F_orig = model_original.F_matrix(dt)
    orig_time = (time.time() - start) / n_iterations
    print(f"Original model: {orig_time:.6f} seconds per call")
    
    start = time.time()
    for _ in range(n_iterations):
        F_restr = model_restructured.F_matrix(dt)
    restr_time = (time.time() - start) / n_iterations
    print(f"Restructured model: {restr_time:.6f} seconds per call")
    print(f"Speedup: {orig_time/restr_time:.2f}x")
    
    # Test Q_matrix performance
    print("\nTesting Q_matrix performance:")
    start = time.time()
    for _ in range(n_iterations):
        Q_orig = model_original.Q_matrix(dt)
    orig_time = (time.time() - start) / n_iterations
    print(f"Original model: {orig_time:.6f} seconds per call")
    
    start = time.time()
    for _ in range(n_iterations):
        Q_restr = model_restructured.Q_matrix(dt)
    restr_time = (time.time() - start) / n_iterations
    print(f"Restructured model: {restr_time:.6f} seconds per call")
    print(f"Speedup: {orig_time/restr_time:.2f}x")
    
    # Run profiling on restructured model
    print("\nProfiling restructured model:")
    results = model_restructured.profile_performance(dt, 5)
    print(f"F_matrix time: {results['F_matrix_time']:.6f} seconds")
    print(f"Q_matrix time: {results['Q_matrix_time']:.6f} seconds")
    
    return model_original, model_restructured

def test_partitioned_filter_single_timestep():
    """Test the PartitionedKalmanFilter class using preprocessed data for a single timestep."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Take just the first two timesteps
    single_timestep_residuals = processed_pulsar_residuals[0:2]
    
    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    print("Initializing the model")
    model = models.SGWB_restructured(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    # Implement the predict method for testing
    def predict_method(self, dt):
        """Standard predict implementation for testing."""
        F = self.model.F_matrix(dt)
        Q = self.model.Q_matrix(dt)
        
        self.xp = F @ self.x
        self.Pp = F @ self.P @ F.T + Q
    
    # Patch the predict method
    kalman_filter.PartitionedKalmanFilter.predict = predict_method

    print("Initialize the Partitioned Kalman filter")
    KF = kalman_filter.PartitionedKalmanFilter(
        model=model, 
        observations=single_timestep_residuals,
        x0=x0, 
        P0=P0
    )

    print("Set global params")
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    print("Run single timestep")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    
    print(f"Log-likelihood for single timestep: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    return KF

def test_compare_partitioned_vs_standard():
    """Compare performance between standard and partitioned Kalman filters."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")
    
    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Take a subset of the data for testing
    test_residuals = processed_pulsar_residuals[:100]
    
    # Get the HD correlation matrix
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    # Initialize the model
    model = models.SGWB_restructured(pulsar_metadata, hd_correlation_matrix)

    # Initialize state vectors
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12
    
    # Set parameters
    params = {
        "γa": 1e-1,
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20,
        "f0": 100 * np.ones(len(pulsar_metadata)),
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }
    
    # Implement optimized predict method for testing
    def optimized_predict(self, dt):
        """Optimized predict implementation for testing."""
        # This is a placeholder - you'll implement your optimized version
        F = self.model.F_matrix(dt)
        Q = self.model.Q_matrix(dt)
        
        self.xp = F @ self.x
        self.Pp = F @ self.P @ F.T + Q
    
    # Patch the predict method
    kalman_filter.PartitionedKalmanFilter.predict = optimized_predict
    
    # Initialize standard filter
    KF_std = kalman_filter.ScalarKalmanFilter(
        model=model, 
        observations=test_residuals,
        x0=x0, 
        P0=P0
    )
    
    # Initialize partitioned filter
    KF_part = kalman_filter.PartitionedKalmanFilter(
        model=model, 
        observations=test_residuals,
        x0=x0, 
        P0=P0
    )
    
    # Test standard filter
    print("\nRunning standard filter:")
    start = time.time()
    ll_std = KF_std.get_likelihood(params)
    std_time = time.time() - start
    print(f"Log-likelihood: {ll_std}")
    print(f"Time taken: {std_time:.4f} seconds")
    
    # Test partitioned filter
    print("\nRunning partitioned filter:")
    start = time.time()
    ll_part = KF_part.get_likelihood(params)
    part_time = time.time() - start
    print(f"Log-likelihood: {ll_part}")
    print(f"Time taken: {part_time:.4f} seconds")
    print(f"Speedup: {std_time/part_time:.2f}x")
    
    # Check that the likelihoods are the same (within numerical precision)
    assert np.isclose(ll_std, ll_part, rtol=1e-10), "Likelihoods don't match!"
    
    return KF_std, KF_part

