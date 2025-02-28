from argus import data_loader, models,gravitational_waves
import os
import glob
import numpy as np


def test_StochasticGWBackgroundModel():
    """Test the StochasticGWBackgroundModel class by loading data, initializing the model, setting parameters, and verifying matrix shapes."""
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
    tim_files = sorted(glob.glob(os.path.join(directory, "*.tim")))

    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    # Instead of manually merging dataframes and computing angles, use the new function.
    # Select the first J pulsars 
    J = 3
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            par_files[0:J], tim_files[0:J]
        )
    )

    # Also get the separation angles between all pulsars.
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)
    # Post-process the residuals
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)
    
    print("Total length of the data is ", len(processed_pulsar_residuals))
    print("Total number of pulsars is ", len(pulsar_metadata))


    # Initialize the GW background model
    model = models.StochasticGWBackgroundModel(pulsar_metadata, hd_correlation_matrix,pulsar_design_matrices)



    # Set global parameters.
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

    model.set_global_parameters(params)

    dt = 0.50
    F = model.F_matrix(dt)
    Q = model.Q_matrix(dt)
    H = model.H_matrix(psr_idx=0)
    R = model.R_matrix(1e-10,0)

    assert F.shape == (model.nx, model.nx)
    assert Q.shape == F.shape
    #more good tests needed here