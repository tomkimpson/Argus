
import jax
import jax.numpy as jnp


def check_cholesky(matrix,matrix_name="Matrix"):
  """Attempts Cholesky decomposition. Returns True if successful, False otherwise."""

  try:
    # Attempt Cholesky decomposition
    L = jnp.linalg.cholesky(matrix)


    # Check for NaNs or infinities in result
    has_nan = jnp.any(jnp.isnan(L))
    has_inf = jnp.any(jnp.isinf(L))
    all_zero = jnp.all(L == 0.0)
    success = (~has_nan) & (~has_inf) & (~all_zero)

    # Print result
    def print_success():
        jax.debug.print("📊 {name} Cholesky: Successful ✅", name=matrix_name,ordered=True)

    def print_failure():
        jax.debug.print("📊 {name} Cholesky: Failed (NaN, inf, or all-zero result) ❌. Has nan: {has_nan}, has inf: {has_inf}, all zero: {all_zero}", name=matrix_name,has_nan=has_nan,has_inf=has_inf,all_zero=all_zero,ordered=True)


    jax.lax.cond(success, print_success, print_failure)

  except ValueError as e:
      # jax.linalg.cholesky raises ValueError for non-positive definite matrices
      # Note: Catching specific errors like this works outside jit,
      # but handling errors *inside* jit often requires different JAX patterns (e.g., jnp.where).
      # For debugging purposes using jax.debug.print, this structure is okay.
      jax.debug.print("⚠️ {matrix_name} Cholesky Decomposition: Failed ❌ (Likely not positive definite)",matrix_name=matrix_name,ordered=True)
  


def check_min_eigenvalue(matrix, matrix_name="Matrix"):
  """Calculates the minimum eigenvalue using eigvalsh (assumes matrix is symmetric)."""
  # Use eigvalsh for numerically symmetric matrices - it's faster and guarantees real eigenvalues.
  # If the matrix might be non-symmetric due to errors, eigvals might be needed,
  # but the goal here is usually to check deviation from the theoretical symmetric positive definite state.
  try:
      eigenvalues = jnp.linalg.eigvalsh(matrix)
      min_eig = jnp.min(eigenvalues)

      def print_warning():
          jax.debug.print("📊 {name} min eigenvalue below threshold ❌, with: {val} < -{tol}", name=matrix_name, val=min_eig, tol=tol)

      def print_success():
          jax.debug.print("📊 {name} minimum eigenvalue is positive ✅ or only slightly negative, with: {val}",name=matrix_name, val=min_eig,ordered=True) 


      tol = 1e-12
      too_negative = min_eig < -tol
      jax.lax.cond(too_negative, print_warning, print_success)


  except Exception as e:
      # Catch potential errors during eigenvalue computation, e.g., non-convergence
      # This might happen if the matrix is severely ill-conditioned or non-symmetric
      jax.debug.print("⚠️ Error computing eigenvalues for {matrix_name}",matrix_name=matrix_name,ordered=True)


def check_symmetry(matrix, matrix_name="Matrix"):
  """Calculates the Frobenius norm of the difference between a matrix and its transpose."""

  diff = matrix - matrix.T
  norm_diff = jnp.linalg.norm(diff, ord='fro')
  # Use jax.debug.print for JAX compatibility inside jit
  jax.debug.print("📊 {matrix_name} Symmetry Error (Frobenius Norm): {norm_diff}",matrix_name=matrix_name, norm_diff=norm_diff,ordered=True)
  return norm_diff



def check_condition_number(matrix, matrix_name="Matrix"):
  """Calculates the condition number."""
  try:
      cond_num = jnp.linalg.cond(matrix)
      # Use jax.debug.print
      jax.debug.print("📊 {matrix_name} Condition Number: {cond_num}",matrix_name=matrix_name,cond_num=cond_num,ordered=True)
      return cond_num
  except Exception as e:
      # Catch potential errors, e.g., for singular matrices
      jax.debug.print("⚠️ Error computing condition number for {matrix_name}: {e}",matrix_name=matrix_name,e=e,ordered=True)
      return jnp.nan # Return NaN or Inf might be appropriate