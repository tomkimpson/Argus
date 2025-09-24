"""Preprocessing script to generate test data for CI environments.

This script processes pulsar timing data locally (where TEMPO2 is available) and saves
the processed results to be used by unit tests in CI environments where TEMPO2
cannot be easily installed.

Run this script manually when the underlying IPTA dataset changes or when setting up
the test data for the first time.

Usage:
    python test/get_pulsar_data_for_testing.py
"""

import os
import pickle
import sys
from pathlib import Path

# Add the python directory to the path to import argus modules
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
python_dir = project_root / "python"
sys.path.insert(0, str(python_dir))

from argus import data_loader


def main():
    """Generate preprocessed pulsar data for testing."""

    # Define paths
    script_dir = Path(__file__).parent.absolute()
    data_directory = script_dir / "../data/IPTA_MockDataChallenge2/dataset_2b/"
    output_file = script_dir / "data/processed_pulsar_data.pkl"

    # Validate input directory
    if not data_directory.exists():
        print(f"Error: Data directory not found: {data_directory}")
        print("Please ensure the IPTA MockDataChallenge2 dataset is available.")
        return 1

    if not data_directory.is_dir():
        print(f"Error: Path is not a directory: {data_directory}")
        return 1

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing pulsar data from: {data_directory}")
    print(f"Output file: {output_file}")

    try:
        # Process the pulsar data using the same parameters as the test
        # Exclude J1640+2224 as in the original test
        pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
            str(data_directory),
            excluded_psrs=['J1640+2224']
        )

        # Save the processed data to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(pulsar_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Successfully processed and saved pulsar data.")
        print(f"   - Processed residuals shape: {pulsar_data['processed_residuals']['toas'].shape}")
        print(f"   - Number of pulsars: {len(pulsar_data['metadata'])}")
        print(f"   - Design matrices: {len(pulsar_data['design_matrices'])}")
        print(f"   - HD correlation matrix shape: {pulsar_data['hd_correlation'].shape}")

        # Display some basic info about the processed data
        print("\nPulsar metadata preview:")
        if 'metadata' in pulsar_data and not pulsar_data['metadata'].empty:
            print(pulsar_data['metadata'][['psr', 'RA', 'DEC']].head())

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: Required files not found: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Error: Data processing failed: {e}")
        return 1
    except ImportError as e:
        print(f"❌ Error: Missing required modules: {e}")
        print("Make sure enterprise-pulsar and TEMPO2 are properly installed.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)