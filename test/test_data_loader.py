from argus import data_loader
import os
import glob
import pytest
import random
import pandas as pd
import numpy as np


def test_load_MDC1_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the invariant directory path
    directory = os.path.join(script_dir, "../data/data_from_enterprise/mdc1/")

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(os.path.join(directory, "*.par")))
    tim_files = sorted(glob.glob(os.path.join(directory, "*.tim")))

    assert len(par_files) == len(
        tim_files
    ), "Mismatch between number of .par and .tim files."

    # Combine par_files and tim_files into pairs and select 5 random pairs
    file_pairs = list(zip(par_files, tim_files))
    random_pairs = random.sample(file_pairs, 5)

    # Check we can load the files with no errors
    for par_file, tim_file in random_pairs:
        try:
            psr = data_loader.LoadWidebandPulsarData.read_par_tim(par_file, tim_file)
        except Exception as e:
            pytest.fail(
                f"Failed to load pulsar data from {par_file} and {tim_file} with error: {e}"
            )


def test_load_NANOGrav15_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the invariant directory path
    directory = os.path.join(script_dir, "../data/NANOGrav15year/wideband/")

    # Get all .par and .tim files in their respective subdirectories
    par_files = sorted(glob.glob(os.path.join(directory, "par/*.par")))
    tim_files = sorted(glob.glob(os.path.join(directory, "tim/*.tim")))

    assert len(par_files) == len(
        tim_files
    ), "Mismatch between number of .par and .tim files."

    # Combine par_files and tim_files into pairs and select 5 random pairs
    file_pairs = list(zip(par_files, tim_files))
    random_pairs = random.sample(file_pairs, 5)

    # Check we can load the files with no errors, using alternative timing package options.
    for par_file, tim_file in random_pairs:
        try:
            psr = data_loader.LoadWidebandPulsarData.read_par_tim(
                par_file,
                tim_file,
                timing_package="pint",
                ephem="DE440",
                bipm_version="BIPM2019",
                clk="TT(BIPM2019)",
            )
        except Exception as e:
            pytest.fail(
                f"Failed to load pulsar data from {par_file} and {tim_file} with error: {e}"
            )


def test_load_MDC1_multiple_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the invariant directory path for MDC1 data
    directory = os.path.join(script_dir, "../data/data_from_enterprise/mdc1/")

    par_files = sorted(glob.glob(os.path.join(directory, "*.par")))
    tim_files = sorted(glob.glob(os.path.join(directory, "*.tim")))

    assert len(par_files) == len(
        tim_files
    ), "Mismatch between number of .par and .tim files."

    # Combine into pairs and sample up to 3 pairs (or all if less than 3)
    file_pairs = list(zip(par_files, tim_files))
    n_samples = min(3, len(file_pairs))
    sample_pairs = random.sample(file_pairs, n_samples)

    sample_par_files = [pair[0] for pair in sample_pairs]
    sample_tim_files = [pair[1] for pair in sample_pairs]

    try:
        merged_df, meta_df = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            sample_par_files, sample_tim_files
        )
    except Exception as e:
        pytest.fail(f"Failed to load multiple MDC1 pulsar data with error: {e}")

    # Check that merged_df is a DataFrame with a 'toas' column
    assert isinstance(merged_df, pd.DataFrame), "merged_df is not a pandas DataFrame."
    assert "toas" in merged_df.columns, "merged_df does not contain a 'toas' column."

    # Check that meta_df is a DataFrame with required columns
    required_meta_cols = {"name", "dim_M", "RA", "DEC"}
    assert isinstance(meta_df, pd.DataFrame), "meta_df is not a pandas DataFrame."
    assert required_meta_cols.issubset(
        set(meta_df.columns)
    ), "meta_df is missing required columns."

    # # Check that angle_matrix is a NumPy array and has shape (N, N)
    # assert isinstance(angle_matrix, np.ndarray), "angle_matrix is not a numpy array."
    # n_pulsars = len(meta_df)
    # assert angle_matrix.shape == (n_pulsars, n_pulsars), "angle_matrix does not have the correct shape."


def test_load_NANOGrav15_multiple_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the invariant directory path for NANOGrav15 data
    directory = os.path.join(script_dir, "../data/NANOGrav15year/wideband/")

    par_files = sorted(glob.glob(os.path.join(directory, "par/*.par")))
    tim_files = sorted(glob.glob(os.path.join(directory, "tim/*.tim")))

    assert len(par_files) == len(
        tim_files
    ), "Mismatch between number of .par and .tim files."

    # Combine into pairs and sample up to 3 pairs (or all if less than 3)
    file_pairs = list(zip(par_files, tim_files))
    n_samples = min(3, len(file_pairs))
    sample_pairs = random.sample(file_pairs, n_samples)

    sample_par_files = [pair[0] for pair in sample_pairs]
    sample_tim_files = [pair[1] for pair in sample_pairs]

    try:
        merged_df, meta_df = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            sample_par_files,
            sample_tim_files,
            timing_package="pint",
            ephem="DE440",
            bipm_version="BIPM2019",
            clk="TT(BIPM2019)",
        )
    except Exception as e:
        pytest.fail(f"Failed to load multiple NANOGrav15 pulsar data with error: {e}")

    # Check that merged_df is a DataFrame with a 'toas' column
    assert isinstance(merged_df, pd.DataFrame), "merged_df is not a pandas DataFrame."
    assert "toas" in merged_df.columns, "merged_df does not contain a 'toas' column."

    # Check that meta_df is a DataFrame with required columns
    required_meta_cols = {"name", "dim_M", "RA", "DEC"}
    assert isinstance(meta_df, pd.DataFrame), "meta_df is not a pandas DataFrame."
    assert required_meta_cols.issubset(
        set(meta_df.columns)
    ), "meta_df is missing required columns."

    # # Check that angle_matrix is a NumPy array and has shape (N, N)
    # assert isinstance(angle_matrix, np.ndarray), "angle_matrix is not a numpy array."
    # n_pulsars = len(meta_df)
    # assert angle_matrix.shape == (n_pulsars, n_pulsars), "angle_matrix does not have the correct shape."
