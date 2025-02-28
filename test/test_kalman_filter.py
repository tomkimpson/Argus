import os
import glob
from argus import data_loader, models, kalman_filter,gravitational_waves
import numpy as np
import pandas as pd
import cProfile
import pstats
import time



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
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals
    processed_pulsar_residuals = (
        data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)
    )

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
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

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
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

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
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

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
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

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
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

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


