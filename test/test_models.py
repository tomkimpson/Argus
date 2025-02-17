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
    merged_df, combined_df, ζ = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
        par_files[0:2], tim_files[0:2]
    )

    print(combined_df)

    # Initialize the GW background model with the metadata dataframe.
    model = models.StochasticGWBackgroundModel(combined_df)

    # Set global parameters.
    params = {
        'γa': 0.001,                    # s⁻¹
        'γp': np.ones(len(combined_df)),
        'σp': 1e-10 * np.ones(len(combined_df)),
        'h2': 1e-12,
        'σeps': 1,
        'separation_angle_matrix': ζ,
        'f0': np.ones(len(combined_df)),
        'σt': 1e-1
    }

    model.set_global_parameters(params)
   
    dt = 0.50
    F = model.F_matrix(dt)
    Q = model.Q_matrix(dt)
    # H = model.H_matrix()
    # R = model.R_matrix()

    assert F.shape == (model.nx, model.nx)
    assert Q.shape == F.shape
