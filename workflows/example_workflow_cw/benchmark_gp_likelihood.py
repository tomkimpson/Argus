"""Benchmark enterprise GP likelihood evaluation for direct comparison with KF.

Loads the same IPTA MDC2 dataset 3b and measures single likelihood eval time
using enterprise's standard GP signal model.
"""

import glob
import time
import numpy as np
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base, white_signals, gp_signals, utils
from enterprise.signals import parameter as ent_parameter
from enterprise import constants as const

# Load pulsars from the same dataset
data_path = "../../data/IPTA_MockDataChallenge2/dataset_3b/"  # relative from workflow dir
# But we need absolute path for reliability
import os
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../data/IPTA_MockDataChallenge2/dataset_3b/")
data_path = os.path.normpath(data_path)

par_files = sorted(glob.glob(os.path.join(data_path, "*.par")))
tim_files = sorted(glob.glob(os.path.join(data_path, "*.tim")))

excluded = ["J1640+2224"]

print(f"Loading pulsars from {data_path}")
print(f"Found {len(par_files)} par files")

pulsars = []
for par, tim in zip(par_files, tim_files):
    psr_name = os.path.basename(par).replace(".par", "")
    if psr_name in excluded:
        continue
    try:
        psr = Pulsar(par, tim, drop_t2pulsar=False)
        pulsars.append(psr)
    except Exception as e:
        print(f"  Skipping {psr_name}: {e}")

print(f"\nLoaded {len(pulsars)} pulsars")
total_toas = sum(len(p.toas) for p in pulsars)
print(f"Total TOAs: {total_toas}")
for p in pulsars[:3]:
    print(f"  {p.name}: {len(p.toas)} TOAs")
print(f"  ...")

# Build enterprise signal model (standard CW search setup)
# White noise (EFAC + EQUAD) + red noise (power-law GP) per pulsar
print("\n=== Building Enterprise Signal Model ===")

# Number of frequency components for red noise GP
n_freqs = 30

t_model_start = time.time()

model_list = []
for psr in pulsars:
    # White noise (EFAC only, enterprise v3.3+ API)
    wn = white_signals.MeasurementNoise(efac=ent_parameter.Constant(1.0))

    # Red noise (power-law, GP basis)
    rn = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw(
            log10_A=ent_parameter.Constant(-15.0),
            gamma=ent_parameter.Constant(3.0),
        ),
        components=n_freqs,
    )

    # Timing model
    tm = gp_signals.TimingModel()

    # Combine
    model = tm + wn + rn
    model_list.append(model(psr))

# Create PTA object
pta = signal_base.PTA(model_list)
t_model_end = time.time()
print(f"Model setup time: {t_model_end - t_model_start:.2f}s")
print(f"Number of parameters: {len(pta.params)}")

# Get default parameter values
x0 = np.array([p.sample() for p in pta.params])
print(f"Parameter vector length: {len(x0)}")

# Benchmark likelihood evaluation
print(f"\n=== Enterprise GP Likelihood Benchmark ({n_freqs} freq components) ===")
print(f"Dataset: {len(pulsars)} pulsars, {total_toas} total TOAs")

# JIT/warmup
print("Warming up...")
for _ in range(5):
    ll = pta.get_lnlikelihood(x0)
print(f"Likelihood value: {ll:.2f}")

# Benchmark
for N in [10, 100, 500]:
    t0 = time.time()
    for _ in range(N):
        ll = pta.get_lnlikelihood(x0)
    t_total = time.time() - t0
    ms_per = t_total / N * 1000
    print(f"{N:5d} evals: {t_total:.3f}s total, {ms_per:.3f} ms/eval, {N/t_total:.1f} evals/sec")

# Also try with different numbers of frequency components
print(f"\n=== Scaling with Number of Frequency Components ===")
for nf in [10, 30, 50, 100]:
    model_list_nf = []
    for psr in pulsars:
        wn = white_signals.MeasurementNoise(efac=ent_parameter.Constant(1.0))
        rn = gp_signals.FourierBasisGP(
            spectrum=utils.powerlaw(
                log10_A=ent_parameter.Constant(-15.0),
                gamma=ent_parameter.Constant(3.0),
            ),
            components=nf,
        )
        tm = gp_signals.TimingModel()
        model = tm + wn + rn
        model_list_nf.append(model(psr))

    pta_nf = signal_base.PTA(model_list_nf)
    x_nf = np.array([p.sample() for p in pta_nf.params])

    # Warmup
    for _ in range(3):
        pta_nf.get_lnlikelihood(x_nf)

    N = 100
    t0 = time.time()
    for _ in range(N):
        pta_nf.get_lnlikelihood(x_nf)
    t_total = time.time() - t0
    ms_per = t_total / N * 1000
    print(f"  n_freq={nf:3d}: {ms_per:.3f} ms/eval ({N/t_total:.1f} evals/sec)")
