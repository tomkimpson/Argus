"""Diagnose the NUTS warmup step-size collapse seen on MDC2 2b (chain 1, every rung) and on
the OU self-generated spike dataset (all four chains).

For each run it rebuilds the exact model the run used (same config, same kernel settings,
same seed), then reports:

  1. INIT: each chain's starting point, reproduced from the run's seed exactly as
     MCMC.run does it (split the seed key per chain, HMC.init on each). Potential energy,
     gradient norm, finiteness, the largest-gradient sites, and the initial step size NUTS's
     own heuristic finds there.
  2. FROZEN: each chain's final unconstrained position from the run's checkpoint, with the
     adapted step size and inverse mass matrix. Same potential/gradient report, plus the
     one-leapfrog energy error across step sizes 1e-14 .. 1 -- the quantity dual averaging
     reacts to. A smooth potential gives |dH| ~ eps^2 (or eps^3); a flat |dH| that does not
     shrink with eps means numerical roughness, not curvature.

Read-only: nothing in the runs is modified.

Usage (CPU; run as a SLURM job, not on the login node):
    JAX_PLATFORMS=cpu python scripts/diag_warmup_collapse.py \
        configs/mdc2_flat_eps000.ini outputs/mdc2_flat_eps000/mdc2_flat_eps000_checkpoint.pkl
"""

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python")
sys.path.insert(0, os.path.abspath(REPO_PY))

from jax import random  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro.infer import NUTS  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402

from argus import bayesian_inference, io_manager, prior_models, utils, workflow  # noqa: E402

EPS_GRID = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def build(config_path):
    config = utils.load_config(config_path)
    config = utils.resolve_config_paths(config, config_path)
    logger = io_manager.setup_single_logger(config, enable_file_logging=False)
    pulsar_data, KF = workflow.setup_data_and_kalman_filter(config, logger, True)
    efac, equad, sigma_p, gamma_p = utils.get_noise_parameters(config)
    n_psr = len(pulsar_data["metadata"])
    names = [str(n) for n in pulsar_data["metadata"]["name"]]
    specs = prior_models.get_prior_model_specs(
        config, n_psr, sigma_p, gamma_p, efac, equad, mode="gwb"
    )

    def model():
        return bayesian_inference.numpyro_model(KF, specs, n_psr)

    kernel = NUTS(
        model,
        target_accept_prob=config.getfloat("NUTS", "target_accept_prob", fallback=0.95),
        max_tree_depth=config.getint("NUTS", "max_tree_depth", fallback=10),
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=config.getboolean("NUTS", "dense_mass", fallback=False),
    )
    seed = config.getint("NUTS", "seed", fallback=42)
    n_chains = config.getint("NUTS", "num_chains", fallback=2)
    n_warm = config.getint("NUTS", "num_warmup", fallback=2000)
    pe_fn = (
        initialize_model(random.PRNGKey(0), model).potential_fn)
    return kernel, seed, n_chains, n_warm, names, pe_fn, KF


def grad_report(pe_fn, z, label, names):
    pe, g = jax.value_and_grad(pe_fn)(z)
    flat_g, _ = ravel_pytree(g)
    flat_g = np.asarray(flat_g)
    print(f"  {label}: PE = {float(pe):.3f}  finite PE: {bool(np.isfinite(pe))}  "
          f"finite grad: {bool(np.all(np.isfinite(flat_g)))}  "
          f"|grad| = {np.linalg.norm(flat_g):.3e}")
    rows = []
    for site, v in g.items():
        v = np.atleast_1d(np.asarray(v))
        for i, x in enumerate(v):
            tag = f"{site}[{i}]" if v.size > 1 else site
            if v.size == len(names) and v.size > 1:
                tag += f" ({names[i]})"
            rows.append((abs(x) if np.isfinite(x) else np.inf, tag, x))
    rows.sort(reverse=True)
    for mag, tag, x in rows[:6]:
        print(f"      grad {tag:<40s} {x: .3e}")
    return float(pe), g


def leapfrog_dH(pe_fn, z, inv_mass, mass_sqrt, key):
    """One leapfrog step from z with fresh momentum, using NumPyro's own integrator and
    the chain's adapted (possibly block-structured) mass matrix; dH per step size."""
    from numpyro.infer.hmc_util import euclidean_kinetic_energy, velocity_verlet
    from numpyro.infer.hmc import momentum_generator

    r = momentum_generator(z, mass_sqrt, key)
    vv_init, vv_update = velocity_verlet(pe_fn, euclidean_kinetic_energy)
    st = vv_init(z, r)
    h0 = float(st.potential_energy + euclidean_kinetic_energy(inv_mass, st.r))
    out = []
    for eps in EPS_GRID:
        st1 = vv_update(eps, inv_mass, st)
        h1 = float(st1.potential_energy + euclidean_kinetic_energy(inv_mass, st1.r))
        out.append((eps, h1 - h0))
    return out


def mass_report(inv_mass):
    """Summarise a (possibly block) inverse mass matrix: per block, diagonal range and
    the eigenvalue range (a zero/negative eigenvalue means a degenerate metric)."""
    blocks = inv_mass.items() if isinstance(inv_mass, dict) else [("all", inv_mass)]
    for site, m in blocks:
        m = np.asarray(m)
        if m.ndim == 1:
            print(f"      inv-mass {str(site)[:60]:<60s} diag  [{m.min():.2e}, {m.max():.2e}]"
                  f"  n<=0: {(m <= 0).sum()} / {m.size}")
        else:
            ev = np.linalg.eigvalsh(m)
            print(f"      inv-mass {str(site)[:60]:<60s} dense eig [{ev.min():.2e}, {ev.max():.2e}]"
                  f"  diag {np.round(np.diag(m), 8)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint", nargs="?")
    args = ap.parse_args()

    print(f"===== {args.config} =====")
    kernel, seed, n_chains, n_warm, names, pe_fn, _ = build(args.config)
    keys = random.split(random.PRNGKey(seed), n_chains)

    print(f"\n--- INIT (seed {seed}, {n_chains} chains, init_to_uniform) ---")
    for c in range(n_chains):
        st = kernel.init(keys[c], n_warm, None, (), {})
        post = kernel.postprocess_fn((), {})(st.z)
        print(f" chain {c}: heuristic initial step size = {float(st.adapt_state.step_size):.3e}")
        for k in ("log10_pivot_psd", "log10_ha", "log10_gamma_a"):
            if k in post:
                print(f"      init {k} = {float(post[k]):.3f}")
        sp = np.asarray(post.get("log10_σp", np.nan))
        gp = np.asarray(post.get("log10_γp", np.nan))
        print(f"      init log10_σp range [{sp.min():.2f}, {sp.max():.2f}]  "
              f"log10_γp range [{gp.min():.2f}, {gp.max():.2f}]")
        grad_report(pe_fn, st.z, "init", names)

    if not args.checkpoint:
        return
    print(f"\n--- FROZEN / FINAL states from {args.checkpoint} ---")
    ck = pickle.load(open(args.checkpoint, "rb"))
    s = ck["sampler_state"]
    for c in range(n_chains):
        z = jax.tree.map(lambda a: jnp.asarray(a)[c], s.z)
        step = float(np.asarray(s.adapt_state.step_size)[c])
        inv_m = jax.tree.map(lambda a: jnp.asarray(a)[c], s.adapt_state.inverse_mass_matrix)
        m_sqrt = jax.tree.map(lambda a: jnp.asarray(a)[c], s.adapt_state.mass_matrix_sqrt)
        post = kernel.postprocess_fn((), {})(z)
        piv = float(post["log10_pivot_psd"]) if "log10_pivot_psd" in post else np.nan
        print(f" chain {c}: adapted step = {step:.3e}  pivot = {piv:.3f}")
        mass_report(inv_m)
        grad_report(pe_fn, z, "final", names)
        for eps, dh in leapfrog_dH(pe_fn, z, inv_m, m_sqrt, random.PRNGKey(c)):
            print(f"      eps {eps:8.0e}  dH = {dh: .4e}")

if __name__ == "__main__":
    main()
