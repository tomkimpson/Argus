from src import data_loader, models 
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
        '../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/'
    )

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + '*.par'))
    tim_files = sorted(glob.glob(os.path.join(directory, '*.tim')))

    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    # Instead of manually merging dataframes and computing angles, use the new function.
    # Select the first 2 file pairs.
    pulsar_residuals, pulsar_metadata = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files[0:2], tim_files[0:2])


    # Also get the separation angles between all pulsars.
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra,dec)


    # Post-process the residuals
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)

    # Initialize the GW background model 
    model = models.StochasticGWBackgroundModel(pulsar_metadata)

    # Initialize the GW background model with the metadata dataframe.
    model = models.StochasticGWBackgroundModel(pulsar_metadata)

    # Set global parameters.
    params = {
        'γa': 1e-1,                    # s⁻¹
        'γp': 1e-1*np.ones(len(pulsar_metadata)),
        'σp': 1e-20 * np.ones(len(pulsar_metadata)),
        'h2': 1e-12,
        'σeps': 1e-20,
        'separation_angle_matrix': angular_separation_matrix,
        'f0': 100*np.ones(len(pulsar_metadata)), #everything is 100 Hz for now
        'EFAC': np.ones(len(pulsar_metadata)),
        'EQUAD':np.ones(len(pulsar_metadata))
    }

    model.set_global_parameters(params)
   
    dt = 0.50
    F = model.F_matrix(dt)
    Q = model.Q_matrix(dt)
    # H = model.H_matrix()
    # R = model.R_matrix()

    assert F.shape == (model.nx, model.nx)
    assert Q.shape == F.shape
