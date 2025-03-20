import os
import glob
from argus import data_loader, models, functional_kalman_filter, gravitational_waves
import numpy as np
import pandas as pd
import cProfile
import pstats
import time


def test_filter_run():
    """Test the functional Kalman filter by loading data and running the filter."""
 
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
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    )

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)

    print("Total length of the data is ", len(processed_pulsar_residuals))
    print("Total number of pulsars is ", len(pulsar_metadata))

    # We still need the model to get dimensions and some parameters, but won't use it in the filter
    print("Initializing the model for dimensions")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix, pulsar_design_matrices)

    # Get dimensions
    Npsr = len(pulsar_metadata)
    M_sum = model.M_sum



    print(pulsar_design_matrices[0].dtype)
    import sys
    sys.exit()
    # Initialize state vectors
    x_gw0   = np.zeros((2*Npsr)) 
    x_spin0 = np.zeros((2*Npsr))
    x_eps0  = np.zeros((M_sum))
    x0_list = [x_gw0, x_spin0, x_eps0]

    # Initialize covariance matrices
    P_gw0   = np.zeros((2*Npsr, 2*Npsr))
    P_spin0 = np.zeros((2*Npsr, 2*Npsr))
    P_eps0  = np.zeros((M_sum, M_sum))
    
    P_gw_spin0 = np.zeros((2*Npsr, 2*Npsr))
    P_gw_eps0 = np.zeros((2*Npsr, M_sum))
    P_spin_eps0 = np.zeros((2*Npsr, M_sum))
    
    P0_list = [P_gw0, P_spin0, P_eps0, P_gw_spin0, P_gw_eps0, P_spin_eps0]

    # Set parameters
    params = {
        "γa": 1e-1,  # s⁻¹
        "γp": 1e-1 * np.ones(Npsr),
        "σp": 1e-20 * np.ones(Npsr),
        "h2": 1e-12,
        "σeps": 1e-20 * np.ones(M_sum),
        "f0": 100 * np.ones(Npsr),  # everything is 100 Hz for now
        "EFAC": np.ones(Npsr),
        "EQUAD": np.ones(Npsr),
    }

    # Extract required data from processed residuals
    data = processed_pulsar_residuals[:, 1]  # measurements
    data_errors = processed_pulsar_residuals[:, 2]  # measurement errors
    psr_indices = processed_pulsar_residuals[:, 3].astype(int)  # pulsar indices
    t_diffs = np.diff(processed_pulsar_residuals[:, 0])  # time differences

    # Initialize design matrix counter
    design_matrix_counter = np.zeros(Npsr).astype(int)

    print("Running the functional filter")   
    # Create profiler and profile the get_likelihood call
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    print("first run")
    ll = functional_kalman_filter.get_likelihood(
        θ=params,
        data=data,
        data_errors=data_errors,
        psr_indices=psr_indices,
        t_diffs=t_diffs,
        pulsar_design_matrices=pulsar_design_matrices,
        design_matrix_counter=design_matrix_counter,
        M_cumsum=model.M_cumsum,
        f0=params["f0"],
        x0_list=x0_list,
        P0_list=P0_list,
        Npsr=Npsr,
        M_sum=M_sum
    )
    
    print("second run")
    ll = functional_kalman_filter.get_likelihood(
        θ=params,
        data=data,
        data_errors=data_errors,
        psr_indices=psr_indices,
        t_diffs=t_diffs,
        pulsar_design_matrices=pulsar_design_matrices,
        design_matrix_counter=design_matrix_counter,
        M_cumsum=model.M_cumsum,
        f0=params["f0"],
        x0_list=x0_list,
        P0_list=P0_list,
        Npsr=Npsr,
        M_sum=M_sum
    )
    
    # profiler.disable()
    print(f"Log-likelihood: {ll}")
    
    # # Print profiling results with more detail
    # print("\nDetailed Performance Profile:")
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(20)
    # stats.sort_stats('tottime').print_stats(20)

if __name__ == "__main__":
    test_filter_run() 