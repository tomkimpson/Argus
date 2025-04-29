
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









# def check_min_eigenvalue(matrix, matrix_name="Matrix"):
#   """Calculates the minimum eigenvalue using eigvalsh (assumes matrix is symmetric)."""
#   # Use eigvalsh for numerically symmetric matrices - it's faster and guarantees real eigenvalues.
#   # If the matrix might be non-symmetric due to errors, eigvals might be needed,
#   # but the goal here is usually to check deviation from the theoretical symmetric positive definite state.
#   try:
#       eigenvalues = jnp.linalg.eigvalsh(matrix)
#       min_eig = jnp.min(eigenvalues)

#       def print_warning():
#           jax.debug.print("📊 {name} min eigenvalue below threshold ❌, with: {val} < -{tol}", name=matrix_name, val=min_eig, tol=tol)

#       def print_success():
#           jax.debug.print("📊 {name} minimum eigenvalue is positive ✅ or only slightly negative, with: {val}",name=matrix_name, val=min_eig,ordered=True) 


#       tol = 1e-12
#       too_negative = min_eig < -tol
#       jax.lax.cond(too_negative, print_warning, print_success)


#   except Exception:
#       # Catch potential errors during eigenvalue computation, e.g., non-convergence
#       # This might happen if the matrix is severely ill-conditioned or non-symmetric
#       jax.debug.print("⚠️ Error computing eigenvalues for {matrix_name}",matrix_name=matrix_name,ordered=True)












# def check_cholesky(matrix,matrix_name="Matrix"):
#   """Attempts Cholesky decomposition. Returns True if successful, False otherwise."""
#   try:
#     # Attempt Cholesky decomposition
#     L = jnp.linalg.cholesky(matrix)


#     # Check for NaNs or infinities in result
#     has_nan = jnp.any(jnp.isnan(L))
#     has_inf = jnp.any(jnp.isinf(L))
#     all_zero = jnp.all(L == 0.0)
#     success = (~has_nan) & (~has_inf) & (~all_zero)

#     # Print result
#     def print_success():
#         jax.debug.print("📊 {name} Cholesky: Successful ✅", name=matrix_name,ordered=True)

#     def print_failure():
#         jax.debug.print("📊 {name} Cholesky: Failed (NaN, inf, or all-zero result) ❌. Has nan: {has_nan}, has inf: {has_inf}, all zero: {all_zero}", name=matrix_name,has_nan=has_nan,has_inf=has_inf,all_zero=all_zero,ordered=True)


#     jax.lax.cond(success, print_success, print_failure)

#   except ValueError:
#       # jax.linalg.cholesky raises ValueError for non-positive definite matrices
#       # Note: Catching specific errors like this works outside jit,
#       # but handling errors *inside* jit often requires different JAX patterns (e.g., jnp.where).
#       # For debugging purposes using jax.debug.print, this structure is okay.
#       jax.debug.print("⚠️ {matrix_name} Cholesky Decomposition: Failed ❌ (Likely not positive definite)",matrix_name=matrix_name,ordered=True)
  








# def check_symmetry(matrix, matrix_name="Matrix"):
#   """Calculates the Frobenius norm of the difference between a matrix and its transpose."""
#   diff = matrix - matrix.T
#   norm_diff = jnp.linalg.norm(diff, ord='fro')
#   # Use jax.debug.print for JAX compatibility inside jit
#   jax.debug.print("📊 {matrix_name} Symmetry Error (Frobenius Norm): {norm_diff}",matrix_name=matrix_name, norm_diff=norm_diff,ordered=True)
#   return norm_diff



# def check_condition_number(matrix, matrix_name="Matrix"):
#   """Calculates the condition number."""
#   try:
#       cond_num = jnp.linalg.cond(matrix)
#       # Use jax.debug.print
#       jax.debug.print("📊 {matrix_name} Condition Number: {cond_num}",matrix_name=matrix_name,cond_num=cond_num,ordered=True)
#       return cond_num
#   except Exception as e:
#       # Catch potential errors, e.g., for singular matrices
#       jax.debug.print("⚠️ Error computing condition number for {matrix_name}: {e}",matrix_name=matrix_name,e=e,ordered=True)
#       return jnp.nan # Return NaN or Inf might be appropriate