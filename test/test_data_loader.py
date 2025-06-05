from argus import data_loader
import os
import glob
import pytest
import random
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def data_files():
    """Load the data files for the IPTA second mock data challenge."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the invariant directory path
    directory = os.path.join(script_dir, "../data/IPTA_MockDataChallenge2/dataset_2b/")

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(os.path.join(directory, "*.par")))
    tim_files = sorted(glob.glob(os.path.join(directory, "*.tim")))
    
    # Ensure we have some files to test with
    if not par_files or not tim_files:
        pytest.skip(f"No .par or .tim files found in {directory}")
        
    return par_files, tim_files

def test_load_individual_pulsars(data_files):
    """Test that we can load each pulsar individually in the IPTA second mock data challenge."""
    par_files, tim_files = data_files

    assert len(par_files) == len(tim_files), "Mismatch between number of .par and .tim files."

    # Combine par_files and tim_files into pairs and select 5 random pairs
    file_pairs = list(zip(par_files, tim_files))
    random_pairs = random.sample(file_pairs, 5)

    # Check we can load the files individually with no errors
    for par_file, tim_file in random_pairs:
        try:
            _ = data_loader.LoadWidebandPulsarData.read_par_tim(par_file, tim_file)
        except Exception as e:
            pytest.fail(
                f"Failed to load pulsar data from {par_file} and {tim_file} with error: {e}"
            )

def test_load_multiple_pulsars(data_files):
    """Test that we can load the multiple pulsars from the IPTA second mock data challenge."""
    par_files, tim_files = data_files

    #Check we can load the multiple files with no errors
    try:
        pulsar_data_frames, pulsar_metadata, design_matrices, parameter_covariances = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    except Exception as e:
        pytest.fail(f"Failed to load multiple MDC2 pulsar data with error: {e}")

    # Basic validation
    assert len(pulsar_data_frames) == len(par_files)
    assert len(pulsar_metadata) == len(par_files)
    assert len(design_matrices) == len(par_files)
    assert len(parameter_covariances) == len(par_files)
    
    # Check that each DataFrame has the expected columns
    for df in pulsar_data_frames:
        assert 'toas' in df.columns
        assert 'residuals' in df.columns
        assert 'error' in df.columns
    
    # Check metadata DataFrame structure
    expected_columns = ['name', 'dim_M', 'RA', 'DEC', 'F0', 'par_file', 'tim_file']
    for col in expected_columns:
        assert col in pulsar_metadata.columns

def test_read_multiple_par_tim_data_consistency(data_files):
    """Test consistency between individual reads and batch read."""
    par_files, tim_files = data_files
    
    # Use just the first file for simplicity
    single_par = par_files[0]
    single_tim = tim_files[0]
    
    # Read individual file
    single_psr = data_loader.LoadWidebandPulsarData.read_par_tim(single_par, single_tim)
    f0 = data_loader.LoadWidebandPulsarData.get_par_value(single_par, 'F0')
    
    # Create expected DataFrame 
    expected_df = pd.DataFrame({
        "toas": single_psr.toas,
        "residuals": single_psr.residuals,
        "error": single_psr.toaerrs
    })
    
    # Read in batch mode
    result = data_loader.LoadWidebandPulsarData.read_multiple_par_tim([single_par], [single_tim])
    pulsar_dfs, metadata_df, design_matrices, param_covariances = result
    
    # Compare results
    pd.testing.assert_frame_equal(pulsar_dfs[0], expected_df)
    assert metadata_df['name'].iloc[0] == single_psr.name
    assert metadata_df['F0'].iloc[0] == f0
    assert np.array_equal(design_matrices[0], single_psr.M_scaled)
    assert np.array_equal(param_covariances[0], single_psr.P_eps)

def test_read_multiple_par_tim_mismatched_lengths(data_files):
    """Test for ValueError when par_files and tim_files have different lengths."""
    par_files, tim_files = data_files
    
    with pytest.raises(ValueError):
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files[:-1])

def test_process_residuals_integration(data_files):
    """Test integration with process_pulsar_residuals_by_epoch."""
    par_files, tim_files = data_files
        
    # Load pulsars
    pulsar_dfs, metadata_df, _, _ = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    
    # Process residuals
    result = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(pulsar_dfs)
    avg_toas = result['toas']
    residuals_array = result['residuals']
    errors_array = result['errors']
    
    # Validate output shapes
    assert len(avg_toas) == len(pulsar_dfs[0])  # Should match number of TOAs
    assert residuals_array.shape == (len(pulsar_dfs[0]), len(pulsar_dfs))
    assert errors_array.shape == (len(pulsar_dfs[0]), len(pulsar_dfs))


    #Explicitly check the output of the process_pulsar_residuals_by_epoch function
    toas_arrays = [df['toas'].values for df in pulsar_dfs]
    
    # Stack arrays into a 2D array where each column is a pulsar's TOAs
    toas_matrix = np.column_stack(toas_arrays)
    
    # Calculate mean manually using numpy
    expected_avg_toas = np.mean(toas_matrix, axis=1)
    
    # Verify average TOAs match expected calculation
    np.testing.assert_array_almost_equal(avg_toas, expected_avg_toas,decimal=5)
    
    # Calculate variance at each epoch
    toa_std_days = np.std(toas_matrix, axis=1)/(3600*24)   
    assert np.all(toa_std_days <= 3)


    #Check edge cases are handled correctly

    #If the input list of DataFrames is empty, we should raise a ValueError
    with pytest.raises(ValueError):
        empty_pulsar_dfs = []
        _, _, _ = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(empty_pulsar_dfs)

    #pop out a column from one of the DataFrames
    #check that the function raises a ValueError
    with pytest.raises(ValueError):
        pulsar_dfs[0].pop('toas')
        _, _, _ = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(pulsar_dfs)



