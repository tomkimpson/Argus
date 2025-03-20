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
    # x0 = np.zeros(model.nx)
    # P0 = np.eye(model.nx) * 1e-12

    x_gw0   = np.zeros((2*model.Npsr)) 
    x_spin0 = np.zeros((2*model.Npsr))
    x_eps0  = np.zeros((model.M_sum))
    x0 = [x_gw0, x_spin0, x_eps0]

    #Initialise the covariance matrices
    P_gw0   = np.zeros((2*model.Npsr, 2*model.Npsr))
    P_spin0 = np.zeros((2*model.Npsr, 2*model.Npsr))
    P_eps0  = np.zeros((model.M_sum, model.M_sum))
    
    P_gw_spin0 = np.zeros((2*model.Npsr, 2*model.Npsr))
    P_gw_eps0 = np.zeros((2*model.Npsr, model.M_sum))
    P_spin_eps0 = np.zeros((2*model.Npsr, model.M_sum))
    
    P0 = [P_gw0, P_spin0, P_eps0, P_gw_spin0, P_gw_eps0, P_spin_eps0]


    KF = kalman_filter.ScalarKalmanFilter(model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0)

    # # Set global parameters. In an inference run we will search for the best parameters.
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(len(pulsar_metadata)),
        "σp": 1e-20 * np.ones(len(pulsar_metadata)),
        "h2": 1e-12,
        "σeps": 1e-20 * np.ones(model.M_sum),
        "f0": 100 * np.ones(len(pulsar_metadata)),  # everything is 100 Hz for now
        "EFAC": np.ones(len(pulsar_metadata)),
        "EQUAD": np.ones(len(pulsar_metadata)),
    }

    print("Running the filter")   
    # Create profiler and profile only the get_likelihood call
    profiler = cProfile.Profile()
    profiler.enable()
    print("first run")
    ll = KF.get_likelihood(params)
    print("second run")
    ll = KF.get_likelihood(params)
    profiler.disable()
    print(f"Log-likelihood: {ll}")
    
    # Print profiling results with more detail
    print("\nDetailed Performance Profile:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)
    stats.sort_stats('tottime').print_stats(20)




