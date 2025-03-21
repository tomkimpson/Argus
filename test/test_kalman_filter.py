import os
import glob
from argus import data_loader, models, kalman_filter,gravitational_waves
import numpy as np
import pandas as pd
import cProfile
import pstats
import time
import sys 

def test_filter_run():
    """Test the KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""
    # Generate some data
    # Load some data to test on
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
    pulsar_residuals, pulsar_metadata,pulsar_design_matrices = (data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files))

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals. 
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)

    print("Total length of the data is ", len(processed_pulsar_residuals))
    print("Total number of pulsars is ", len(pulsar_metadata))

    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix,pulsar_design_matrices)



    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    KF = kalman_filter.ScalarKalmanFilter(model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0)



    # # Set global parameters. In an inference run we will search for the best parameters.
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20 * np.ones(model.M_sum),
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

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


