import os
import glob
from argus import data_loader, models, kalman_filter,gravitational_waves
import numpy as np
import pandas as pd
import cProfile
import pstats
import time
import sys 
from flax import struct
import jax.numpy as jnp
import jax 


jax.config.update('jax_log_compiles', True) #show when compiles 
jax.config.update("jax_enable_x64", True)   # Enforce 64 bit precision


def _get_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    data_path = "../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/"
    directory = os.path.join(
        script_dir,
        data_path
    )

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))
    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    # Get the data
    print(f"Getting the data. Loading {len(par_files)} pulsars from {data_path}")
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices = (data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files))

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)

    print("Total length of the data is ", len(processed_pulsar_residuals))
    print("Total number of pulsars is ", len(pulsar_metadata))

    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)
    
    return model, processed_pulsar_residuals, pulsar_metadata

def test_filter():
    """Test the KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""
    
    # Get the data and model
    model, processed_pulsar_residuals, pulsar_metadata = _get_data()

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    KF = kalman_filter.ScalarKalmanFilter(model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0)
 
    # Create a struct for the parameters
    @struct.dataclass
    class ModelParameters:
        γa: float
        γp: np.ndarray
        σeps: np.ndarray
        σp: np.ndarray
        EFAC: np.ndarray 
        EQUAD: np.ndarray

    Npsr = len(pulsar_metadata)
    M_sum = model.M_sum
    params = ModelParameters(γa=1e-1, 
                             γp=1e-1 * np.ones(Npsr), #arbitrary 
                             σeps=1e-20 * np.ones(M_sum), #arbitrary
                             σp=1e-20 * np.ones(Npsr), #arbitrary
                             EFAC=1e-20 * np.ones(Npsr), #arbitrary
                             EQUAD=1e-20 * np.ones(Npsr) #arbitrary
    )


    print("Running the filter")
    
    # Set up cProfile
    profiler = cProfile.Profile()
    start_time = time.time()
    
    profiler.enable()
    ll = KF.get_likelihood(params)
    profiler.disable()
    stats = pstats.Stats(profiler)

    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")



    print("Running again")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    print(f"Time taken for round 2: {end_time - start_time:.4f} seconds")







    # Filter and print stats for your modules only
    print("\n--- Profiling Results ---")
    stats.sort_stats(pstats.SortKey.TIME)
    # Filter to only include your modules
    stats.print_stats('argus')  # This will include all functions from the argus package
    
    #Run it one more time
    #Model must be reinitialised to get around the H indexig
    # model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix,pulsar_design_matrices)
    # KF = kalman_filter.ScalarKalmanFilter(model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0)
    # ll = KF.get_likelihood(params)












def test_jax_filter():
    """Test the JAX-based KalmanFilter implementation."""
    
    # Get the data and model
    model, processed_pulsar_residuals, pulsar_metadata = _get_data()

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    # Use the JAX implementation
    KF = kalman_filter.JaxKalmanFilter(model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0)

    # Create parameter struct
    @struct.dataclass
    class KalmanParams:
        γa: float
        γp: jnp.ndarray
        σeps: jnp.ndarray
        σp: jnp.ndarray

    Npsr = len(pulsar_metadata)
    M_sum = model.M_sum
    params = KalmanParams(γa=1e-1, γp=1e-1 * np.ones(Npsr), σeps=1e-20 * np.ones(M_sum), σp=1e-20 * np.ones(Npsr))

    print("Running the JAX filter")
    
    # Set up cProfile
    profiler = cProfile.Profile()
    start_time = time.time()
    
    profiler.enable()
    ll = KF.get_likelihood(params)
    profiler.disable()
    stats = pstats.Stats(profiler)

    end_time = time.time()
    
    print(f"Log-likelihood: {ll}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    print("Running again")
    start_time = time.time()
    ll = KF.get_likelihood(params)
    end_time = time.time()
    print(f"Time taken for round 2: {end_time - start_time:.4f} seconds")

    # Filter and print stats for your modules only
    print("\n--- Profiling Results ---")
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats('argus')  # This will include all functions from the argus package


