# Diffuse (Exact) Kalman Filtering for PTA Timing-Model Marginalization
This note summarizes the mathematics and provides JAX code for an **exact diffuse Kalman filter (DKF)** update that *removes the timing-model parameters from the state* while preserving the per-epoch, time-recursive structure required by a Kalman filter. It is tailored to the **PTA** setting and is **equivalent** to the standard **GLS/SVD marginalization** of timing-model parameters (van Haasteren & Levin) **conditional on the white-noise covariance** (which depends on EFAC/EQUAD).

---
## 1) PTA observation model and blocks
For each epoch (stacked across pulsars), the measurement equation is
$$
y_k = H_k x_k + M_k \beta + e_k, \quad e_k \sim \mathcal N(0, R_k),
$$
where:
- $x_k$ collects the **stochastic** latent states (GW and spin blocks only).
- $H_k$ is the measurement matrix from those stochastic blocks to timing residuals.
- $M_k$ contains the **timing-model regressors** (rows of the per-pulsar design matrices) at epoch $k$.
- $\beta$ are the **deterministic** timing-model coefficients (RA/DEC, DM terms, etc.).
- $R_k$ is the per-epoch measurement-noise covariance, **built from EFAC/EQUAD** (and any other white-noise components you include).

The **state dynamics** follow your block model:
$$
x_{k+1} = F_k x_k + \eta_k, \quad \eta_k \sim \mathcal N(0, Q_k),
$$
with OU-style blocks in both GW and spin components.

We wish to **marginalize $\beta$** under a **diffuse (improper flat) prior** *without* breaking the sequential KF structure.

---
## 2) Why not a global GLS projection for a KF?
The classic GLS/SVD marginalization projects the **stacked** data by a left projector $Z^\top$ that depends on $R$ and $M$. That mixes *times*, so the projected datum at step $k$ is a linear combination of measurements from multiple epochs. A standard KF update (which consumes one epoch at a time) is no longer applicable.

Instead we use the **diffuse Kalman filter** (Durbin–Koopman). It gives a **per-epoch** recursion that is *exactly equivalent* to GLS marginalization (conditional on $R$), but keeps time order and preserves the KF’s prediction–update structure.

---
## 3) Diffuse KF for deterministic regressors (per-epoch, whitened)
Define the **whitening** factor (Cholesky) for the epoch:
$$
R_k = L_k L_k^\top, \quad y_k^w = L_k^{-1}(y_k - H_k x_{k|k-1}), \quad H_k^w = L_k^{-1} H_k, \quad M_k^w = L_k^{-1} M_k.
$$
Working in whitened space, the measurement noise is identity.

Maintain across time the **accumulated GLS information** about $\beta$:
$$
\mathcal{J}_{\beta,k} = \sum_{i\le k} (M_i^w)^\top M_i^w, \quad c_{\beta,k} = \sum_{i\le k} (M_i^w)^\top y_i^w.
$$
(We carry $\mathcal{J}_{\beta,k}^{1/2}$ via its Cholesky for numerical stability.)

At epoch $k$, compute:
1. **Stochastic innovation covariance** (with $\beta$ absent) in whitened space
$$
S_{x,k} = H_k^w P_{k|k-1} (H_k^w)^\top + I.
$$
2. **Accumulate GLS information**
$$
\mathcal{J}_{\beta,k} = \mathcal{J}_{\beta,k-1} + (M_k^w)^\top M_k^w, \quad c_{\beta,k} = c_{\beta,k-1} + (M_k^w)^\top y_k^w.
$$
3. **Solve for the GLS posterior mode** of $\beta$ (exact, square-root)
$$
\hat\beta_k = \mathcal{J}_{\beta,k}^{-1} c_{\beta,k}.
$$
4. **Diffuse-marginalized innovation and covariance**
$$
\tilde y_k = y_k^w - M_k^w \hat\beta_k, \quad \tilde S_k = S_{x,k} + M_k^w \, \mathcal{J}_{\beta,k}^{-1} \, (M_k^w)^\top.
$$
The second term in $\tilde S_k$ is the **variance increase** that accounts for integrating out $\beta$ under a diffuse prior. This yields the **exact** (per-epoch) prediction-error contribution for the timing-model–marginalized likelihood.

5. **Kalman update for the stochastic state**
$$
K_{x,k} = P_{k|k-1} (H_k^w)^\top \tilde S_k^{-1}, \quad x_{k|k} = x_{k|k-1} + K_{x,k} \tilde y_k,
$$
$$
P_{k|k} = (I - K_{x,k} H_k^w) P_{k|k-1} (I - K_{x,k} H_k^w)^\top + K_{x,k} R^{\text{eff}}_k K_{x,k}^\top,
$$
with $R^{\text{eff}}_k = \tilde S_k - H_k^w P_{k|k-1} (H_k^w)^\top = I + M_k^w \, \mathcal{J}_{\beta,k}^{-1} \, (M_k^w)^\top$.
The **Joseph form** above is numerically robust.

6. **Log-likelihood contribution (prediction-error decomposition)**
In whitened space,
$$
\log p(y_k | \cdot) = -\tfrac12\big(\log|\tilde S_k| + \tilde y_k^\top \tilde S_k^{-1} \tilde y_k + n_y \log 2\pi\big) - \tfrac12 \log|L_k|^2.
$$
Summing over $k$ gives the full marginalized log-likelihood.

> **Equivalence to GLS**: Accumulating $\mathcal{J}_{\beta,k}$ and $c_{\beta,k}$ across all epochs and using the above $\tilde y_k, \tilde S_k$ reproduces the van Haasteren–Levin GLS/SVD marginalized likelihood (Eqs. (22)/(25)) **conditional on $R=\operatorname{diag}((\mathrm{EFAC}\cdot\sigma)^2+\mathrm{EQUAD}^2)$**.

---
## 4) EFAC/EQUAD and rebuilding the whitening
If EFAC/EQUAD are **fixed**, the procedure is exact with a single whitening per epoch. If EFAC/EQUAD are **inferred**, you must rebuild the Cholesky $L_k$ (hence $M_k^w,H_k^w,y_k^w$) for each proposed EFAC/EQUAD setting.

---
## 5) JAX implementation (drop-in update)
Below is a self-contained JAX implementation of the exact diffuse update in square-root form.

```python
import jax
import jax.numpy as jnp

def _solve_psd_chol(L, b):
    # Solve (L L^T) x = b via Cholesky factors.
    return jax.scipy.linalg.cho_solve((L, False), b)

def _quadform_inv_chol(L, B):
    # Compute B (LL^T)^{-1} B^T without forming the inverse explicitly.
    X = jax.scipy.linalg.cho_solve((L, False), B.T)  # solve (LL^T) X = B^T
    return B @ X

def _logdet_from_chol(L):
    # Return log|LL^T| from a Cholesky factor L (lower triangular).
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

def _whiten_by_chol(L, Y):
    # Return L^{-1} Y by forward substitution; Y can be (ny,1), (ny,nx), etc.
    return jax.scipy.linalg.solve_triangular(L, Y, lower=True)

def _log_likelihood(y: jax.Array, cov: jax.Array) -> jax.Array:
    # Standard Gaussian log-likelihood: y ~ N(0, cov).
    sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * cov)
    quadratic_term = y.T @ jnp.linalg.solve(cov, y)
    return -0.5 * (logdet + quadratic_term)

def _update_diffuse_exact(xp: jax.Array,
                          Pp: jax.Array,
                          H: jax.Array,            # (ny, nx)  (no timing columns)
                          R: jax.Array,            # (ny, ny)
                          z: jax.Array,            # (ny, 1)
                          M_row: jax.Array,        # (ny, p)   timing regressors at this epoch
                          Rbeta_chol: jax.Array,   # (p, p)    chol of accumulated M^T R^{-1} M
                          c_beta: jax.Array        # (p, 1)    accumulated M^T R^{-1} residuals
                          ):
    """
    Exact diffuse-KF update for deterministic regressors:
      y = H x + M beta + e,   beta ~ diffuse (flat),  e ~ N(0, R).

    Works in whitened space (R -> I). Uses Cholesky (square-root) forms.
    Returns:
      x, P, Rbeta_chol_new, c_beta_new, y_tilde, S_tilde, ll_contrib
    """
    # 1) Whiten once per epoch
    L = jnp.linalg.cholesky(R)                  # R = L L^T
    r = z - H @ xp                              # pre-fit residual vs stochastic state
    y_w = _whiten_by_chol(L, r)                 # (ny,1)
    H_w = _whiten_by_chol(L, H)                 # (ny,nx)
    M_w = _whiten_by_chol(L, M_row)             # (ny,p)

    # 2) Innovation covariance from stochastic state in whitened space
    Sx = H_w @ Pp @ H_w.T + jnp.eye(H_w.shape[0])

    # 3) Accumulate GLS information for timing coefficients (square-root form)
    # Recompute small Cholesky; p is small.
    Rbeta_new = (Rbeta_chol @ Rbeta_chol.T) + (M_w.T @ M_w)          # (p,p)
    Rbeta_chol_new = jnp.linalg.cholesky(Rbeta_new + 1e-18*jnp.eye(Rbeta_new.shape[0]))
    c_beta_new = c_beta + (M_w.T @ y_w)                              # (p,1)

    # 4) GLS posterior mode of beta
    beta_hat = _solve_psd_chol(Rbeta_chol_new, c_beta_new)           # (p,1)

    # 5) Diffuse-marginalized innovation and covariance
    y_tilde = y_w - (M_w @ beta_hat)                                 # (ny,1)
    S_add   = _quadform_inv_chol(Rbeta_chol_new, M_w)                # (ny,ny)
    S_tilde = Sx + S_add                                             # (ny,ny)

    # 6) Kalman update for stochastic state (Joseph form)
    Sinv = jnp.linalg.solve(S_tilde, jnp.eye(S_tilde.shape[0]))
    Kx   = Pp @ H_w.T @ Sinv
    x    = xp + (Kx @ y_tilde)
    R_eff = S_tilde - (H_w @ Pp @ H_w.T)                             # = I + M_w J^{-1} M_w^T
    I_KH = jnp.eye(Pp.shape[0]) - Kx @ H_w
    P    = I_KH @ Pp @ I_KH.T + Kx @ R_eff @ Kx.T

    # 7) Log-likelihood contribution (whitened, include Jacobian of whitening)
    signS, logdetS = jnp.linalg.slogdet(S_tilde)
    quad = (y_tilde.T @ jnp.linalg.solve(S_tilde, y_tilde))[0,0]
    ll = -0.5 * (logdetS + quad + H_w.shape[0] * jnp.log(2.0*jnp.pi)) - 0.5 * _logdet_from_chol(L)

    return x, P, Rbeta_chol_new, c_beta_new, y_tilde, S_tilde, ll
```

---
## 6) Equivalence to GLS (sketch)
Let $C=N+K_{\text{non-TS}}$ be the covariance excluding the timing-model subspace. Classic GLS marginalization gives
$$
\log p(y\mid \theta_{\text{non-TS}}) = -\tfrac12\big(\log|C| + \log|M^\top C^{-1}M| + y^\top C_0 y\big) + \text{const},
$$
with $C_0 = C^{-1} - C^{-1}M(M^\top C^{-1}M)^{-1}M^\top C^{-1}$. The diffuse KF above reproduces this sequentially via whitening, accumulation of $M^\top R^{-1}M$, and the Schur/Woodbury identities.

---
## 7) Practical checklist
- Remove timing block from the state and from `H`.
- Pass `M_row` per epoch and maintain `(Rbeta_chol, c_beta)` across time.
- Rebuild `R` (hence whitening) whenever EFAC/EQUAD change.
- Sum the per-epoch `ll` to get the total marginalized log-likelihood.
- Use Joseph-form covariance update for stability.
