

import os 
import glob 
from src import data_loader,models,kalman_filter
import numpy as np 

def test_filter_run():


    #Generate some data
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
    tim_files = sorted(glob.glob(directory + '*.tim'))
    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    #Get the dataframes
    merged_df, combined_df, ζ = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files[0:1], tim_files[0:1])



    # Initialize the GW background model with the metadata dataframe.
    model = models.StochasticGWBackgroundModel(combined_df)



    #********** the above below be moved to be part of data loader

    #initialise the kalman filter
   # residuals_data = merged_df



    #massage the residuals data into a proper array form. This should be a standalone. 
    #shoudl also assert that no times occur at the same time. todo

    residual_columns = [col for col in merged_df.columns if col.startswith('residuals_')]
    mask = ~merged_df[residual_columns].isna()   # DataFrame of booleans

    # 3) For each row, find the *position* of the True (non-NaN) column
    #    np.argmax returns the index of the first True in each row.
    idx = np.argmax(mask.values, axis=1)  # This is a NumPy array of shape (Nrows,)

    # 4) Extract the numeric part of the column name. 
    #    E.g. "residuals_3" -> 3
    subscript_list = [int(col.split('_')[-1]) for col in residual_columns]

    # Map each row’s True position to its "residuals_i" subscript
    subscripts = np.array(subscript_list)[idx]

    # 5) Use advanced indexing to get the *actual non-NaN values*.
    row_indices = np.arange(len(merged_df))  # 0,1,2,... up to len(df)-1
    res_values = merged_df[residual_columns].values[row_indices, idx]

    # Finally, stack them into a 2D array:
    #   - Column 0: the non-NaN residual value
    #   - Column 1: the subscript i
    result = np.column_stack([merged_df['toas'].values,res_values, subscripts])



    #********** the above should be moved



    #Not sure where we should define x0 and P0....
    x0 = np.zeros(model.nx)
    P0 = np.eye(model.nx)*1e-12

    
    KF=kalman_filter.KalmanFilter(model=model,observations=result,x0=x0,P0=P0)



    # # Set global parameters.
    params = {
        'γa': 1e-1,                    # s⁻¹
        'γp': 1e-1*np.ones(len(combined_df)),
        'σp': 1e-20 * np.ones(len(combined_df)),
        'h2': 1e-12,
        'σeps': 1e-20,
        'separation_angle_matrix': ζ,
        'f0': 100*np.ones(len(combined_df)), #everything is 100 Hz for now
        'σt': 1e-6 #some approximate reasonable value
    }

    KF.get_likelihood(params)

