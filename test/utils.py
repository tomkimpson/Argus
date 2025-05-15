
import jax
import jax.numpy as jnp



import numpy as np 
def check_cholesky(matrix):
  """
  Checks if a matrix allows Cholesky decomposition (i.e., is positive definite).

  Args:
      matrix: The input matrix (JAX array or compatible).

  Returns:
      bool: True if Cholesky decomposition succeeds, False otherwise.
  """
  try:
    # Check for square matrix explicitly
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        # A non-square matrix cannot be positive definite
        return False


    # Attempt Cholesky decomposition
    L = jnp.linalg.cholesky(matrix)

    has_nan = jnp.any(jnp.isnan(L))
    has_inf = jnp.any(jnp.isinf(L))

    #Determine overall success
    # Must succeed Cholesky *and* result must be finite.
    is_successful_and_finite = not (has_nan or has_inf)
    return is_successful_and_finite

  # Catch the specific error raised for non-positive definite matrices.
  except (np.linalg.LinAlgError, ValueError):
    # Cholesky failed, likely matrix is not positive definite
    return False
  except Exception:
    return False




def check_minimum_eigenvalue(matrix, threshold=-1e-12):
  """
  Checks if the minimum eigenvalue of a Hermitian/symmetric matrix is above a threshold.

  Calculates eigenvalues using jnp.linalg.eigvalsh and compares the minimum
  to the specified threshold. Also checks for non-finite eigenvalues.
  Assumes the input matrix is Hermitian (symmetric if real).

  Args:
      matrix: The input matrix (JAX array or compatible).
      threshold (float): The minimum acceptable value for the smallest eigenvalue.
                         Defaults to -1e-12 

  Returns:
      bool: True if the minimum eigenvalue is finite and strictly greater
            than the threshold, False otherwise.
  """
  try:
  
  # Basic shape check: Must be a square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    # Optional: Explicit check for symmetry/Hermitian property if needed
    # if not jnp.allclose(matrix, matrix.T.conj()):
    #     return False # eigvalsh assumes this property

    # --- Core Check Logic ---
    # 1. Calculate eigenvalues using eigvalsh (for Hermitian/symmetric)
    eigenvalues = jnp.linalg.eigvalsh(matrix)

    # 2. Check the calculated eigenvalues for NaNs or Infs
    has_nan = jnp.any(jnp.isnan(eigenvalues))
    has_inf = jnp.any(jnp.isinf(eigenvalues))

    if has_nan or has_inf:
        return False # Eigenvalue computation resulted in non-finite numbers

    # 3. Find the minimum eigenvalue
    min_eigenvalue = jnp.min(eigenvalues)

    # 4. Compare with the threshold and return boolean result
    return min_eigenvalue > threshold

  except (np.linalg.LinAlgError, ValueError):
    # Eigenvalue decomposition failed (e.g., LinAlgError from eigvalsh)
    # or potentially ValueError from input checks/casting if types are weird.
    return False
  except Exception:
    # Catch any other unexpected errors during the check.
    # Consider logging this during debugging if it happens.
    return False




import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import glob
import json
import pandas as pd

from argus import data_loader
from argus import gravitational_waves


def _get_processed_residuals(directory):
    """Get the processed residuals from the data."""

    # Get all .par and .tim files in the directory
    par_files = sorted(glob.glob(directory + "*.par"))
    tim_files = sorted(glob.glob(directory + "*.tim"))

    assert len(par_files) == len(tim_files), "Mismatch between .par and .tim file counts."

    #Exclude PR J1640+2224 as it has an exponent which is to small for the OU process to be valid
    par_files = [f for f in par_files if "J1640" not in f]
    tim_files = [f for f in tim_files if "J1640" not in f]



    # Get the data
    print(f"Getting the data. Loading {len(par_files)} pulsars from {directory}")
    pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices = (
        data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)
    )

    # Get the separation angles and compute HD correlation
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = gravitational_waves.pairwise_angular_separation(ra, dec)
    hd_correlation_matrix = gravitational_waves.hellings_downs(angular_separation_matrix)

    # Post-process the residuals    
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(pulsar_residuals)

    return processed_pulsar_residuals, pulsar_metadata, pulsar_design_matrices,P_eps_matrices,hd_correlation_matrix

def get_efac_equad_injections():

    # Load the noise parameters from the json file
    with open("../data/IPTA_MockDataChallenge2/group1_psr_noise.json", "r") as f:
        noise_params = json.load(f)

    # Extract EFAC and EQUAD values for each pulsar
    efac_values = []
    equad_values = []

    for psr in noise_params:

        if  "J1640" not in psr:
            efac_values.append(noise_params[psr]["efac"])
            equad_values.append(10**noise_params[psr]["equad"]) # Convert from log10 to linear

    # Convert to JAX arrays
    efac_array = jnp.array(efac_values)
    equad_array = jnp.array(equad_values)


    return efac_array, equad_array

def get_psr_noise_injections():

    df = pd.read_pickle('../notebooks/approximate_spin_injections.pkl')
    condition = df['psr'] != 'J1640+2224'



    # 2. Use the condition to select rows and create a new DataFrame
    df_filtered = df[condition]


    sigma_p_injected = df_filtered['optimal_sigma'].values
    gamma_p_injected = df_filtered['optimal_gamma'].values

    return jnp.array(sigma_p_injected), jnp.array(gamma_p_injected)





