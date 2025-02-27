import os
import glob
from argus import data_loader, models, kalman_filter
import numpy as np

def hellings_downs(θ):
    """Compute the Hellings–Downs function for an angle θ (in radians).
    For θ == 0 the autocorrelation is defined to be 1.
    """
    x = (1 - np.cos(θ)) / 2.0
    return np.where(
        np.isclose(θ, 0.0),
        1.0,
        (3 / 2) * x * np.log(x) - x / 4 + 0.5,
    )

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


import pandas as pd


def test_filter_run_v2():
    """Test the KalmanFilter class by loading data, initializing the model, setting parameters, and running the filter."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "../data/preprocessed_data/")

    # Load preprocessed data
    processed_pulsar_residuals = np.load(
        directory + "IPTA_Challenge1_open_Dataset2_residuals.npy"
    )
    pulsar_metadata = pd.read_parquet(
        directory + "IPTA_Challenge1_open_Dataset2_metadata"
    )

    # Also get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = hellings_downs(angular_separation_matrix)

    print(
        "Total length of the data is ", len(processed_pulsar_residuals)
    )  # Calculate the size of the DataFrame in MB
    print(
        "Total number of pulsars is ", len(pulsar_metadata)
    )  # Calculate the size of the DataFrame in MB

    # Initialize the GW background model with HD correlation
    print("Initializing the model")
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix)

    # Initialize the Kalman Filter
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx) * 1e-12

    print("Initialise the Kalman filter")
    KF = kalman_filter.ScalarKalmanFilter(
        model=model, observations=processed_pulsar_residuals, x0=x0, P0=P0
    )

    print("set global params")
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

    print("Iterate")
    KF.get_likelihood(params)
