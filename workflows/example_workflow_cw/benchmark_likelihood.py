"""Benchmark single KF likelihood evaluation on CPU vs GPU."""

import sys
import logging
import time

# Setup logger
logger = logging.getLogger('argus')
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)

import argus.io_manager as iom
iom._argus_logger = logger
iom.get_argus_logger = lambda: logger

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update('jax_enable_x64', True)

from argus.cw_kalman_filter import CWKalmanFilter
from argus.bayesian_inference import cw_log_likelihood_fn

print(f"JAX version: {jax.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print()

# Build synthetic data matching real problem size
np.random.seed(42)
Npsr = 31
nobs = 183

toas_list = [np.sort(np.random.uniform(0, 1e9, nobs)) for _ in range(Npsr)]
residuals_list = [np.random.normal(0, 1e-7, nobs) for _ in range(Npsr)]
errors_list = [np.full(nobs, 1e-7) for _ in range(Npsr)]

metadata = pd.DataFrame({
    'name': [f'J{i:04d}+0001' for i in range(Npsr)],
    'dim_M': [3] * Npsr,
    'RA': np.random.uniform(0, 2 * np.pi, Npsr),
    'DEC': np.random.uniform(-np.pi / 2, np.pi / 2, Npsr),
    'F0': np.random.uniform(100, 600, Npsr),
})
design_matrices = [np.random.randn(nobs, 3) * 0.01 for _ in range(Npsr)]
P_eps = [np.eye(3) * 0.01 for _ in range(Npsr)]

data = {
    'processed_residuals': {
        'toas': toas_list, 'residuals': residuals_list,
        'errors': errors_list, 'n_obs': np.array([nobs] * Npsr),
    },
    'metadata': metadata,
    'design_matrices': design_matrices,
    'parameter_covariances': P_eps,
}

kf = CWKalmanFilter(data, include_pulsar_term=True, phase_parameterization=True)

# Test parameters
log10_h0 = -13.35
alpha_gw = 4.067
delta_gw = 0.14
log10_f_gw = -8.215
cos_iota = 0.907
psi = 0.646
Phi0 = 0.175
chi = jnp.zeros(Npsr)
log10_gp = jnp.full(Npsr, -8.0)
log10_sp = jnp.full(Npsr, -15.0)
efac = jnp.ones(Npsr)
equad = jnp.full(Npsr, 1e-7)

jit_ll = jax.jit(lambda: cw_log_likelihood_fn(
    kf, log10_h0, alpha_gw, delta_gw, log10_f_gw,
    cos_iota, psi, Phi0, chi, log10_gp, log10_sp, efac, equad,
))

# JIT compile
print('=== JIT Compilation ===')
t0 = time.time()
ll = jit_ll()
jax.block_until_ready(ll)
t_compile = time.time() - t0
print(f'Compilation time: {t_compile:.2f}s')
print(f'Likelihood value: {float(ll):.2f}')

# Warmup
for _ in range(50):
    ll = jit_ll()
    jax.block_until_ready(ll)

# Benchmark
print()
print(f'=== Single Eval Benchmark ({jax.default_backend().upper()}) ===')
print(f'Dataset: {Npsr} pulsars x {nobs} observations = {Npsr * nobs} total TOAs')
print()

for N in [100, 1000, 5000]:
    t0 = time.time()
    for _ in range(N):
        ll = jit_ll()
        jax.block_until_ready(ll)
    t_total = time.time() - t0
    ms_per = t_total / N * 1000
    print(f'{N:5d} evals: {t_total:.3f}s total, {ms_per:.3f} ms/eval, {N/t_total:.0f} evals/sec')

# Also benchmark with gradient (relevant for NUTS/HMC)
print()
print(f'=== Single Eval + Gradient Benchmark ({jax.default_backend().upper()}) ===')

def ll_flat(x):
    return cw_log_likelihood_fn(
        kf, x[0], x[1], x[2], x[3], x[4], x[5], x[6],
        x[7:7+Npsr], x[7+Npsr:7+2*Npsr], x[7+2*Npsr:7+3*Npsr],
        efac, equad,
    )

x0 = jnp.concatenate([
    jnp.array([log10_h0, alpha_gw, delta_gw, log10_f_gw, cos_iota, psi, Phi0]),
    chi, log10_gp, log10_sp,
])

jit_grad = jax.jit(jax.value_and_grad(ll_flat))

# Compile
t0 = time.time()
val, grad = jit_grad(x0)
jax.block_until_ready(grad)
t_grad_compile = time.time() - t0
print(f'Gradient compilation: {t_grad_compile:.2f}s')

# Warmup
for _ in range(50):
    val, grad = jit_grad(x0)
    jax.block_until_ready(grad)

for N in [100, 1000]:
    t0 = time.time()
    for _ in range(N):
        val, grad = jit_grad(x0)
        jax.block_until_ready(grad)
    t_total = time.time() - t0
    ms_per = t_total / N * 1000
    print(f'{N:5d} evals: {t_total:.3f}s total, {ms_per:.3f} ms/eval, {N/t_total:.0f} evals/sec')

# GP comparison calculations
print()
print('=== GP Likelihood Cost Estimates (theoretical) ===')
print()
n_toas_list = [183, 1000, 5000, 10000]
for n in n_toas_list:
    # O(N^3) for Cholesky, assume 1 GFLOP/s effective for dense linear algebra on CPU
    # Cholesky on NxN matrix: ~N^3/3 FLOPs
    flops_no_woodbury = (n ** 3) / 3
    # Woodbury: O(N * n_freq^2) where n_freq ~ 30 frequency components
    n_freq = 30
    flops_woodbury = n * (n_freq ** 2)

    # Assume ~10 GFLOP/s effective throughput (single core, double precision)
    gflops = 10.0
    t_no_woodbury_ms = flops_no_woodbury / (gflops * 1e9) * 1000
    t_woodbury_ms = flops_woodbury / (gflops * 1e9) * 1000

    # For Npsr pulsars (independent, so multiply by Npsr)
    print(f'N_obs = {n:6d} per pulsar ({Npsr} pulsars):')
    print(f'  GP (Cholesky, no tricks):  {t_no_woodbury_ms * Npsr:10.2f} ms/eval  '
          f'[{flops_no_woodbury * Npsr / 1e9:.1f} GFLOP]')
    print(f'  GP (Woodbury, {n_freq} freqs):    {t_woodbury_ms * Npsr:10.2f} ms/eval  '
          f'[{flops_woodbury * Npsr / 1e6:.1f} MFLOP]')
    print(f'  KF (O(N)):                 {0.04 * n / 183 * Npsr * 1000 / Npsr:10.2f} ms/eval  '
          f'[scales linearly from benchmark]')
    print()
