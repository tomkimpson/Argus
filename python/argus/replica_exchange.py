"""Replica exchange MCMC (parallel tempering) for CW inference.

Implements parallel tempering with HMC within-chain proposals using blackjax.
K chains run at different inverse temperatures β_0=1 (cold/posterior) down to
β_{K-1}=β_hot (near prior). Hot chains explore broadly and swap discoveries
to the cold chain via Metropolis-Hastings replica exchange.

This is the standard approach for PTA CW searches (PTMCMCSampler, QuickCW),
adapted to use gradient-based HMC proposals with our JAX-differentiable
Kalman filter likelihood.
"""

import time
import json
import os
import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
import blackjax
import arviz as az
import numpy as np

from argus.tempered_smc import (
    build_parameter_registry,
    build_logprior_fn,
    build_loglikelihood_fn,
    unpack_to_physical,
)


# ---------------------------------------------------------------------------
# Temperature ladder
# ---------------------------------------------------------------------------

def build_temperature_ladder(num_chains, beta_hot=0.01, spacing="geometric"):
    """Build inverse temperature ladder for replica exchange.

    Parameters
    ----------
    num_chains : int
        Number of temperature chains (K).
    beta_hot : float
        Inverse temperature of the hottest chain (default 0.01).
    spacing : str
        "geometric" (default) or "linear".

    Returns
    -------
    jnp.ndarray
        Inverse temperatures, shape (K,), from 1.0 (cold) to beta_hot.
    """
    if num_chains == 1:
        return jnp.array([1.0])
    if spacing == "linear":
        return jnp.linspace(1.0, beta_hot, num_chains)
    else:
        # Geometric: beta_k = beta_hot^(k/(K-1))
        exponents = jnp.linspace(0.0, 1.0, num_chains)
        return beta_hot**exponents


# ---------------------------------------------------------------------------
# NUTS warmup for step size / mass matrix adaptation
# ---------------------------------------------------------------------------

def warmup_with_nuts(rng_key, logdensity_fn, initial_position, num_warmup=500):
    """Run NUTS warmup on the cold chain to discover step size and mass matrix.

    Parameters
    ----------
    rng_key : jnp.ndarray
        PRNG key.
    logdensity_fn : callable
        Cold chain (beta=1) log-density function.
    initial_position : jnp.ndarray
        Starting position, shape (ndim,).
    num_warmup : int
        Number of warmup iterations.

    Returns
    -------
    step_size : float
        Adapted step size for the cold chain.
    inverse_mass_matrix : jnp.ndarray
        Diagonal inverse mass matrix, shape (ndim,).
    final_position : jnp.ndarray
        Position after warmup.
    """
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=0.8,
        progress_bar=True,
    )
    (last_state, parameters), _ = warmup.run(rng_key, initial_position, num_warmup)
    return (
        float(parameters["step_size"]),
        parameters["inverse_mass_matrix"],
        last_state.position,
    )


# ---------------------------------------------------------------------------
# Single-chain HMC step (vmappable across chains)
# ---------------------------------------------------------------------------

def _make_single_chain_hmc_step(logprior_fn, loglikelihood_fn, hmc_kernel,
                                 inverse_mass_matrix, num_integration_steps):
    """Create a vmappable single-chain HMC step function.

    Returns a function with signature:
        (rng_key, position, beta, step_size) -> (new_position, loglik, logprior, acc_rate)
    """
    def step(rng_key, position, beta, step_size):
        def tempered_logdensity(x):
            return logprior_fn(x) + beta * loglikelihood_fn(x)

        state = blackjax.hmc.init(position, tempered_logdensity)
        new_state, info = hmc_kernel(
            rng_key, state, tempered_logdensity,
            step_size, inverse_mass_matrix, num_integration_steps,
        )
        new_ll = loglikelihood_fn(new_state.position)
        new_lp = logprior_fn(new_state.position)
        return new_state.position, new_ll, new_lp, info.acceptance_rate

    return step


# ---------------------------------------------------------------------------
# Swap proposals
# ---------------------------------------------------------------------------

def _propose_swaps(rng_key, positions, loglikelihoods, logpriors, betas, even_swap):
    """Propose and accept/reject swaps between adjacent temperature chains.

    Parameters
    ----------
    rng_key : jnp.ndarray
    positions : jnp.ndarray, shape (K, ndim)
    loglikelihoods : jnp.ndarray, shape (K,)
    logpriors : jnp.ndarray, shape (K,)
    betas : jnp.ndarray, shape (K,)
    even_swap : bool
        If True, propose swaps for even-indexed pairs (0-1, 2-3, ...).

    Returns
    -------
    new_positions, new_loglikelihoods, new_logpriors : updated arrays
    swap_accepted : jnp.ndarray, shape (K-1,), boolean per pair
    """
    K = betas.shape[0]
    num_pairs = K - 1

    # Log acceptance ratios for all adjacent pairs
    delta_beta = betas[1:] - betas[:-1]  # negative (betas decrease)
    delta_logL = loglikelihoods[:-1] - loglikelihoods[1:]
    log_alpha = delta_beta * delta_logL

    # Draw uniform random numbers
    log_u = jnp.log(random.uniform(rng_key, shape=(num_pairs,)))
    accepted = log_u < log_alpha

    # Mask: only process even or odd pairs
    pair_indices = jnp.arange(num_pairs)
    mask = jnp.where(even_swap, pair_indices % 2 == 0, pair_indices % 2 == 1)
    accepted = accepted & mask

    # Apply swaps via jnp.where (no in-place mutation)
    new_positions = positions
    new_lls = loglikelihoods
    new_lps = logpriors

    # Process each pair — unrolled at trace time (K is small, typically 8-16)
    for p in range(num_pairs):
        do_swap = accepted[p]
        i, j = p, p + 1

        pos_i = new_positions[i]
        pos_j = new_positions[j]
        ll_i = new_lls[i]
        ll_j = new_lls[j]
        lp_i = new_lps[i]
        lp_j = new_lps[j]

        new_positions = new_positions.at[i].set(jnp.where(do_swap, pos_j, pos_i))
        new_positions = new_positions.at[j].set(jnp.where(do_swap, pos_i, pos_j))
        new_lls = new_lls.at[i].set(jnp.where(do_swap, ll_j, ll_i))
        new_lls = new_lls.at[j].set(jnp.where(do_swap, ll_i, ll_j))
        new_lps = new_lps.at[i].set(jnp.where(do_swap, lp_j, lp_i))
        new_lps = new_lps.at[j].set(jnp.where(do_swap, lp_i, lp_j))

    return new_positions, new_lls, new_lps, accepted


# ---------------------------------------------------------------------------
# Adaptive temperature ladder (Vousden et al. 2016, MNRAS 455, 1919)
# ---------------------------------------------------------------------------

def _adapt_temperature_ladder(betas_np, swap_rates, adapt_count,
                               kappa_0=1.0, decay_scale=10, target=0.234):
    """Vousden et al. (2016) adaptive temperature ladder update.

    Adjusts inverse temperatures to equalize swap acceptance rates across
    all adjacent pairs. Works by modifying log-spacings in temperature space:
        S_j -> S_j + kappa * (A_j - A_{j+1})
    where S_j = log(T_{j+1} - T_j) and A_j is the swap acceptance rate
    for pair (j, j+1). Boundary condition: A_{K-1} = target.

    Endpoints (beta=1 and beta=beta_hot) are kept fixed via rescaling.

    Parameters
    ----------
    betas_np : np.ndarray, shape (K,)
        Current inverse temperatures, descending from 1.0.
    swap_rates : np.ndarray, shape (K-1,)
        Swap acceptance rates per adjacent pair.
    adapt_count : int
        Number of adaptations so far (controls gain decay).
    kappa_0 : float
        Initial gain factor.
    decay_scale : float
        Controls gain decay: kappa = kappa_0 / (1 + adapt_count / decay_scale).
    target : float
        Target swap acceptance rate (boundary condition and goal).

    Returns
    -------
    np.ndarray, shape (K,)
        Updated inverse temperatures.
    """
    K = len(betas_np)
    if K <= 2:
        return betas_np

    kappa = kappa_0 / (1.0 + adapt_count / decay_scale)

    # Work in temperature space (ascending: T[0]=1, T[K-1]=T_max)
    temps = 1.0 / betas_np
    dT = np.diff(temps)
    S = np.log(np.maximum(dT, 1e-10))

    # Update log-spacings: S[j] += kappa * (A_j - A_{j+1})
    # If A_j < A_{j+1}, gap j is too wide → decrease S[j] → narrows gap → improves acceptance.
    # Boundary: A_{K-1} = target (virtual pair beyond the hottest chain).
    for j in range(len(S)):
        A_curr = swap_rates[j]
        A_next = swap_rates[j + 1] if j + 1 < len(swap_rates) else target
        S[j] += kappa * (A_curr - A_next)

    # Reconstruct temperatures preserving T[0]=1
    new_dT = np.exp(S)
    new_temps = np.zeros(K)
    new_temps[0] = 1.0
    for j in range(K - 1):
        new_temps[j + 1] = new_temps[j] + new_dT[j]

    # Rescale to preserve T_max
    T_max = temps[-1]
    total_span = new_temps[-1] - 1.0
    if total_span > 0:
        scale = (T_max - 1.0) / total_span
        new_temps = 1.0 + (new_temps - 1.0) * scale
    new_temps[0] = 1.0
    new_temps[-1] = T_max

    # Ensure strict monotonicity
    for j in range(1, K):
        if new_temps[j] <= new_temps[j - 1]:
            new_temps[j] = new_temps[j - 1] * 1.001

    return 1.0 / new_temps


# ---------------------------------------------------------------------------
# Main replica exchange loop
# ---------------------------------------------------------------------------

def run_replica_exchange(
    logprior_fn,
    loglikelihood_fn,
    ndim,
    num_chains=8,
    num_samples=5000,
    num_warmup=500,
    num_hmc_steps=10,
    num_integration_steps=20,
    beta_hot=0.01,
    beta_spacing="geometric",
    step_size=-1.0,
    inverse_mass_matrix=None,
    seed=42,
    thin=1,
):
    """Run replica exchange MCMC with HMC within-chain proposals.

    Parameters
    ----------
    logprior_fn : callable
        log_prior(x_flat) -> scalar
    loglikelihood_fn : callable
        log_likelihood(x_flat) -> scalar
    ndim : int
        Parameter space dimension.
    num_chains : int
        Number of temperature chains (K).
    num_samples : int
        Post-warmup samples to collect from the cold chain.
    num_warmup : int
        NUTS warmup iterations for step size adaptation.
    num_hmc_steps : int
        HMC proposals per chain between swap attempts.
    num_integration_steps : int
        Leapfrog steps per HMC proposal.
    beta_hot : float
        Inverse temperature of hottest chain.
    beta_spacing : str
        "geometric" or "linear".
    step_size : float
        Base step size. If <= 0, adapted via NUTS warmup.
    inverse_mass_matrix : jnp.ndarray or None
        Diagonal inverse mass matrix. None = adapted via NUTS warmup.
    seed : int
        Random seed.
    thin : int
        Thinning factor.

    Returns
    -------
    dict
        cold_chain_samples, swap_acceptance_rates, hmc_acceptance_rates,
        betas, step_sizes, wall_time, num_warmup, num_samples
    """
    rng_key = random.PRNGKey(seed)

    # 1. Temperature ladder (adaptive starts from geometric, then adapts online)
    initial_spacing = "geometric" if beta_spacing == "adaptive" else beta_spacing
    betas = build_temperature_ladder(num_chains, beta_hot, initial_spacing)
    print(f"Temperature ladder ({num_chains} chains, {beta_spacing}): {betas}")

    # 2. Initialise positions from prior (Normal(0,1) in unconstrained space)
    rng_key, init_key = random.split(rng_key)
    initial_positions = random.normal(init_key, shape=(num_chains, ndim))

    # 3. Warmup: adapt step size and mass matrix on cold chain
    need_warmup = step_size <= 0 or inverse_mass_matrix is None
    if need_warmup:
        print(f"Running NUTS warmup ({num_warmup} steps) for step size adaptation...")
        rng_key, warmup_key = random.split(rng_key)

        def cold_logdensity(x):
            return logprior_fn(x) + loglikelihood_fn(x)

        t_warmup = time.time()
        adapted_ss, adapted_mm, warmup_pos = warmup_with_nuts(
            warmup_key, cold_logdensity, initial_positions[0], num_warmup,
        )
        print(f"  Warmup complete in {time.time() - t_warmup:.1f}s")
        print(f"  Adapted step size: {adapted_ss:.6f}")

        if step_size <= 0:
            # NUTS-adapted step size is typically too large for fixed-trajectory HMC.
            # NUTS can adapt tree depth to compensate, but HMC has a fixed trajectory.
            # Scale down by ~0.5 to improve acceptance rate.
            step_size = adapted_ss * 0.5
            print(f"  HMC step size (0.5x NUTS): {step_size:.6f}")
        if inverse_mass_matrix is None:
            inverse_mass_matrix = adapted_mm
        # Start cold chain from warmup position
        initial_positions = initial_positions.at[0].set(warmup_pos)
    else:
        if inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(ndim)

    # 4. Per-chain step sizes (hot chains use larger steps)
    step_sizes = step_size * jnp.power(betas, -0.25)
    step_sizes = jnp.minimum(step_sizes, step_size * 10.0)
    print(f"Step sizes: cold={step_size:.6f}, hot={float(step_sizes[-1]):.6f}")

    # 5. Build HMC kernel and vmapped step function
    hmc_kernel = blackjax.hmc.build_kernel()
    single_step = _make_single_chain_hmc_step(
        logprior_fn, loglikelihood_fn, hmc_kernel,
        inverse_mass_matrix, num_integration_steps,
    )

    # 6. Initial log-likelihood and log-prior for all chains
    init_lls = jax.vmap(loglikelihood_fn)(initial_positions)
    init_lps = jax.vmap(logprior_fn)(initial_positions)

    # 7. Define one iteration: multi-step HMC + swap
    # JIT-compiled as a single step, called from a Python loop.
    # This avoids the massive XLA graph from lax.scan over thousands of iterations.
    def one_iteration(rng_key, positions, lls, lps, betas, step_sizes, even_swap):
        # --- HMC steps (vmapped across chains) ---
        def multi_hmc(rng_key, position, beta, step_sz):
            """Run num_hmc_steps sequential HMC proposals for one chain."""
            def scan_body(carry, _):
                rng, pos = carry
                rng, step_key = random.split(rng)
                new_pos, new_ll, new_lp, acc = single_step(
                    step_key, pos, beta, step_sz,
                )
                return (rng, new_pos), acc

            (rng_out, final_pos), acceptances = jax.lax.scan(
                scan_body, (rng_key, position), jnp.arange(num_hmc_steps),
            )
            final_ll = loglikelihood_fn(final_pos)
            final_lp = logprior_fn(final_pos)
            mean_acc = jnp.mean(acceptances)
            return final_pos, final_ll, final_lp, mean_acc

        rng_key, *chain_keys = random.split(rng_key, num_chains + 1)
        chain_keys = jnp.stack(chain_keys)

        new_positions, new_lls, new_lps, mean_accs = jax.vmap(
            multi_hmc, in_axes=(0, 0, 0, 0),
        )(chain_keys, positions, betas, step_sizes)

        # --- Swap proposals (alternating even/odd) ---
        rng_key, swap_key = random.split(rng_key)
        new_positions, new_lls, new_lps, swap_accepted = _propose_swaps(
            swap_key, new_positions, new_lls, new_lps, betas, even_swap,
        )

        return rng_key, new_positions, new_lls, new_lps, mean_accs, swap_accepted

    # 8. JIT-compile a single iteration (small graph, fast compile)
    print(f"JIT-compiling single iteration kernel...")
    print(f"  {num_chains} chains x {num_hmc_steps} HMC steps x {num_integration_steps} leapfrog")
    jit_one_iteration = jax.jit(one_iteration, donate_argnums=(1,))

    total_iterations = num_samples * thin

    # 9. Adaptive temperature ladder settings
    adaptive = (beta_spacing == "adaptive")
    adapt_interval = 100     # adapt every N iterations
    adapt_start = 200        # let chains settle before adapting
    adapt_end = int(total_iterations * 0.5)  # freeze ladder at halfway
    adapt_count = 0
    if adaptive:
        betas_np = np.array(betas)
        window_accepts = np.zeros(num_chains - 1)
        window_attempts = np.zeros(num_chains - 1)
        print(f"  Adaptive ladder enabled (Vousden et al. 2016): "
              f"adapting every {adapt_interval} iters, iters {adapt_start}-{adapt_end}")

    # 10. Run sampling via Python loop (no massive XLA graph)
    positions = initial_positions
    lls = init_lls
    lps = init_lps

    cold_samples_list = []
    acc_list = []
    swap_list = []

    t_start = time.time()
    t_jit = None

    for i in range(total_iterations):
        even_swap = i % 2 == 0
        rng_key, positions, lls, lps, mean_accs, swap_accepted = jit_one_iteration(
            rng_key, positions, lls, lps, betas, step_sizes, even_swap,
        )

        # Record JIT compilation time on first iteration
        if i == 0:
            positions.block_until_ready()
            t_jit = time.time() - t_start
            print(f"  JIT compilation + first iteration: {t_jit:.1f}s")

        # Collect cold chain sample (every thin-th iteration)
        if i % thin == 0:
            cold_samples_list.append(positions[0])

        acc_list.append(mean_accs)
        swap_list.append(swap_accepted)

        # Adaptive ladder update
        if adaptive:
            swap_np = np.array(swap_accepted)
            # Track which pairs were actually attempted this iteration
            if even_swap:
                attempted = np.arange(0, num_chains - 1, 2)
            else:
                attempted = np.arange(1, num_chains - 1, 2)
            window_accepts += swap_np
            window_attempts[attempted] += 1

            if ((i + 1) >= adapt_start and (i + 1) <= adapt_end
                    and (i + 1) % adapt_interval == 0):
                rates = np.where(window_attempts > 0,
                                 window_accepts / window_attempts, 0.234)
                betas_np = _adapt_temperature_ladder(
                    betas_np, rates, adapt_count)
                betas = jnp.array(betas_np)
                step_sizes = step_size * jnp.power(betas, -0.25)
                step_sizes = jnp.minimum(step_sizes, step_size * 10.0)
                adapt_count += 1
                window_accepts[:] = 0
                window_attempts[:] = 0
                if adapt_count <= 5 or adapt_count % 10 == 0:
                    print(f"  Ladder adapt #{adapt_count}: "
                          f"swap_rates={rates.round(3)}")

        # Progress reporting
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (total_iterations - i - 1) / rate if rate > 0 else 0
            print(
                f"  iter {i+1}/{total_iterations} "
                f"({elapsed:.0f}s elapsed, {eta:.0f}s remaining, {rate:.1f} it/s)"
            )

    # Force last computation to complete
    positions.block_until_ready()
    wall_time = time.time() - t_start

    print(f"Sampling complete in {wall_time:.1f}s ({wall_time/60:.1f} min)")

    # 11. Stack results
    cold_samples = jnp.stack(cold_samples_list)
    all_accs = jnp.stack(acc_list)
    all_swaps = jnp.stack(swap_list)

    hmc_acceptance_rates = jnp.mean(all_accs, axis=0)
    swap_acceptance_rates = jnp.mean(all_swaps.astype(jnp.float64), axis=0)

    print(f"HMC acceptance rates: {np.array(hmc_acceptance_rates)}")
    print(f"Swap acceptance rates: {np.array(swap_acceptance_rates)}")

    results = {
        "cold_chain_samples": cold_samples,
        "swap_acceptance_rates": swap_acceptance_rates,
        "hmc_acceptance_rates": hmc_acceptance_rates,
        "betas": betas,
        "step_sizes": step_sizes,
        "wall_time": wall_time,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
    }
    if adaptive:
        results["adaptive"] = True
        results["num_ladder_adaptations"] = adapt_count
    return results


# ---------------------------------------------------------------------------
# ArviZ conversion
# ---------------------------------------------------------------------------

def re_results_to_arviz(cold_chain_samples, registry, n_pulsars):
    """Convert replica exchange cold chain samples to ArviZ InferenceData.

    Parameters
    ----------
    cold_chain_samples : jnp.ndarray
        Shape (num_samples, ndim), unconstrained space.
    registry : ParameterRegistry
    n_pulsars : int

    Returns
    -------
    arviz.InferenceData
    """
    # Unpack all samples to physical space
    physical = jax.vmap(lambda x: unpack_to_physical(x, registry))(cold_chain_samples)

    # Build posterior dict with shape (1, num_samples, ...) — 1 chain
    posterior_dict = {}
    for key, val in physical.items():
        arr = np.array(val)
        if arr.ndim == 1:
            posterior_dict[key] = arr[np.newaxis, :]  # (1, num_samples)
        elif arr.ndim == 2:
            posterior_dict[key] = arr[np.newaxis, :, :]  # (1, num_samples, n_pulsars)

    return az.from_dict(posterior=posterior_dict)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def save_re_diagnostics(re_results, output_dir, output_id):
    """Save replica exchange diagnostics to JSON."""
    diag = {
        "betas": np.array(re_results["betas"]).tolist(),
        "step_sizes": np.array(re_results["step_sizes"]).tolist(),
        "hmc_acceptance_rates": np.array(re_results["hmc_acceptance_rates"]).tolist(),
        "swap_acceptance_rates": np.array(re_results["swap_acceptance_rates"]).tolist(),
        "wall_time": float(re_results["wall_time"]),
        "num_warmup": int(re_results["num_warmup"]),
        "num_samples": int(re_results["num_samples"]),
        "num_chains": int(re_results["num_chains"]),
    }
    if re_results.get("adaptive"):
        diag["adaptive"] = True
        diag["num_ladder_adaptations"] = re_results["num_ladder_adaptations"]
    path = os.path.join(output_dir, f"{output_id}_re_diagnostics.json")
    with open(path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"Saved RE diagnostics to {path}")


def plot_re_diagnostics(re_results, output_dir, output_id):
    """Create replica exchange diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas = np.array(re_results["betas"])
    swap_rates = np.array(re_results["swap_acceptance_rates"])
    hmc_rates = np.array(re_results["hmc_acceptance_rates"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Swap acceptance rates
    pair_labels = [f"{i}-{i+1}" for i in range(len(swap_rates))]
    axes[0].bar(pair_labels, swap_rates, color="steelblue")
    axes[0].set_xlabel("Temperature pair")
    axes[0].set_ylabel("Swap acceptance rate")
    axes[0].set_title("Replica exchange swap rates")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.234, color="red", linestyle="--", alpha=0.5, label="Optimal ~23%")
    axes[0].legend()

    # Panel 2: HMC acceptance rates vs temperature
    axes[1].plot(betas, hmc_rates, "o-", color="steelblue")
    axes[1].set_xlabel(r"Inverse temperature $\beta$")
    axes[1].set_ylabel("HMC acceptance rate")
    axes[1].set_title("Per-chain HMC acceptance")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.65, color="red", linestyle="--", alpha=0.5, label="Target ~65%")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f"{output_id}_re_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RE diagnostic plot to {path}")
