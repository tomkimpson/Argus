#!/usr/bin/env python
"""Standard PTA CW search using ENTERPRISE + enterprise_extensions + PTMCMCSampler.

Runs on the same IPTA MDC2 dataset_3b as Argus for direct comparison.
Supports fixed-noise and sampled-noise modes.

Usage:
    python enterprise_cw_search.py --mode fixed
    python enterprise_cw_search.py --mode sampled
    python enterprise_cw_search.py --mode fixed --n-samples 1000  # quick test
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import (
    deterministic_signals,
    gp_signals,
    parameter as ent_parameter,
    signal_base,
    utils,
    white_signals,
)
from enterprise_extensions.deterministic import CWSignal, cw_delay
from enterprise_extensions.sampler import JumpProposal
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../data/IPTA_MockDataChallenge2/dataset_3b")
)
NOISE_FILE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../data/IPTA_MockDataChallenge2/group1_psr_noise.json")
)
DIST_FILE = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../data/IPTA_MockDataChallenge2/pulsar_distances.json")
)

EXCLUDED_PSRS = ["J1640+2224"]
N_FREQ = 30  # Fourier components for red noise GP


def load_pulsars():
    """Load pulsars from dataset_3b, excluding problematic ones."""
    par_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.par")))
    tim_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.tim")))

    with open(NOISE_FILE) as f:
        noise_params = json.load(f)
    with open(DIST_FILE) as f:
        dist_data = json.load(f)

    pulsars = []
    for par, tim in zip(par_files, tim_files):
        psr_name = os.path.basename(par).replace(".par", "")
        if psr_name in EXCLUDED_PSRS:
            continue
        try:
            psr = Pulsar(par, tim, drop_t2pulsar=False)
            # Inject distances from the JSON file if available
            if psr_name in dist_data:
                d_kpc = dist_data[psr_name]["distance_kpc"]
                psr._pdist = np.array([d_kpc, 0.2 * d_kpc])
            pulsars.append(psr)
        except Exception as e:
            print(f"  Skipping {psr_name}: {e}")

    print(f"Loaded {len(pulsars)} pulsars, {sum(len(p.toas) for p in pulsars)} total TOAs")
    return pulsars, noise_params


def build_cw_signal():
    """Build the CW deterministic signal with priors matching Argus config."""
    cw_wf = cw_delay(
        cos_gwtheta=ent_parameter.Uniform(-1.0, 1.0)("cos_gwtheta"),
        gwphi=ent_parameter.Uniform(0.0, 2 * np.pi)("gwphi"),
        log10_fgw=ent_parameter.Uniform(-9.0, -7.0)("log10_fgw"),
        log10_h=ent_parameter.Uniform(-17.0, -12.0)("log10_h"),
        cos_inc=ent_parameter.Uniform(-1.0, 1.0)("cos_inc"),
        psi=ent_parameter.Uniform(0.0, np.pi)("psi"),
        phase0=ent_parameter.Uniform(0.0, 2 * np.pi)("phase0"),
        log10_mc=np.log10(5e9),
        psrTerm=True,
        evolve=False,
        phase_approx=True,
        tref=0,
    )
    return CWSignal(cw_wf, ecc=False, psrTerm=True)


def build_model(pulsars, noise_params, mode="fixed"):
    """Build the full PTA signal model.

    Parameters
    ----------
    pulsars : list
        Enterprise Pulsar objects.
    noise_params : dict
        Per-pulsar noise injection values.
    mode : str
        'fixed' or 'sampled' noise.
    """
    cw = build_cw_signal()

    model_list = []
    for psr in pulsars:
        psr_name = psr.name
        noise = noise_params.get(psr_name, {})

        # Timing model (marginalizes over timing model parameters)
        tm = gp_signals.TimingModel()

        if mode == "fixed":
            efac_val = noise.get("efac", 1.0)
            equad_val = noise.get("equad", -7.0)  # log10(EQUAD in seconds)
            rn_A = noise.get("rn_log10_A", -15.0)
            rn_gamma = noise.get("rn_spec_ind", 3.0)

            wn = white_signals.MeasurementNoise(
                efac=ent_parameter.Constant(efac_val),
                log10_t2equad=ent_parameter.Constant(equad_val),
            )
            rn = gp_signals.FourierBasisGP(
                spectrum=utils.powerlaw(
                    log10_A=ent_parameter.Constant(rn_A),
                    gamma=ent_parameter.Constant(rn_gamma),
                ),
                components=N_FREQ,
            )
        else:  # sampled
            wn = white_signals.MeasurementNoise(
                efac=ent_parameter.Uniform(0.1, 3.0)(
                    "_".join([psr_name, "efac"])
                ),
                log10_t2equad=ent_parameter.Uniform(-9.0, -5.0)(
                    "_".join([psr_name, "log10_t2equad"])
                ),
            )
            rn = gp_signals.FourierBasisGP(
                spectrum=utils.powerlaw(
                    log10_A=ent_parameter.Uniform(-18.0, -11.0)(
                        "_".join([psr_name, "rn_log10_A"])
                    ),
                    gamma=ent_parameter.Uniform(0.0, 7.0)(
                        "_".join([psr_name, "rn_gamma"])
                    ),
                ),
                components=N_FREQ,
            )

        model = tm + wn + rn + cw
        model_list.append(model(psr))

    pta = signal_base.PTA(model_list)
    return pta


def run_sampler(pta, outdir, n_samples=500_000, thin=10, n_temps=5, t_max=1000):
    """Run PTMCMCSampler with parallel tempering."""
    os.makedirs(outdir, exist_ok=True)

    x0 = np.array([p.sample() for p in pta.params])
    ndim = len(x0)

    # Build geometric temperature ladder
    ladder = np.geomspace(1.0, t_max, n_temps)

    print(f"\nSampler configuration:")
    print(f"  Parameters: {ndim}")
    print(f"  Samples: {n_samples:,}")
    print(f"  Thinning: {thin}")
    print(f"  Temperature ladder: {ladder}")
    for i, p in enumerate(pta.params):
        print(f"  [{i:3d}] {p.name}")

    # Test likelihood at initial point
    ll0 = pta.get_lnlikelihood(x0)
    lp0 = pta.get_lnprior(x0)
    print(f"\nInitial log-likelihood: {ll0:.2f}")
    print(f"Initial log-prior:      {lp0:.2f}")

    cov = np.diag(np.ones(ndim) * 0.01**2)

    sampler = ptmcmc(
        ndim,
        pta.get_lnlikelihood,
        pta.get_lnprior,
        cov,
        outDir=outdir,
        resume=False,
    )

    # Add jump proposals
    jp = JumpProposal(pta)
    sampler.addProposalToCycle(jp.draw_from_prior, 15)
    sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)

    burn = int(n_samples * 0.25)

    print(f"\nStarting sampling at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    t0 = time.time()

    sampler.sample(
        x0,
        n_samples,
        ladder=ladder,
        SCAMweight=30,
        AMweight=15,
        DEweight=50,
        burn=burn,
        thin=thin,
    )

    elapsed = time.time() - t0
    print(f"Sampling complete in {elapsed/3600:.2f} hours")
    return sampler


def postprocess(pta, outdir):
    """Convert PTMCMCSampler chains to ArviZ NetCDF."""
    import arviz as az

    chain_file = os.path.join(outdir, "chain_1.0.txt")
    if not os.path.exists(chain_file):
        print(f"Chain file not found: {chain_file}")
        return

    chain = np.loadtxt(chain_file)
    print(f"Loaded chain: {chain.shape}")

    # Columns: parameters, log-likelihood, log-prior, acceptance, temperature
    param_names = [p.name for p in pta.params]
    n_params = len(param_names)

    # Extract parameter columns (first n_params columns)
    samples = chain[:, :n_params]

    # Enterprise -> Argus parameter name mapping
    name_map = {
        "cos_gwtheta": "cos_gwtheta",  # will compute delta_gw
        "gwphi": "alpha_gw",
        "log10_fgw": "log10_f_gw",
        "log10_h": "log10_h0",
        "cos_inc": "cos_iota",
        "psi": "psi",
        "phase0": "Phi0",
    }

    cw_params = ["log10_h0", "log10_f_gw", "alpha_gw", "delta_gw",
                 "cos_iota", "psi", "Phi0"]

    posterior_dict = {}
    for i, pname in enumerate(param_names):
        argus_name = name_map.get(pname, pname)
        posterior_dict[argus_name] = samples[:, i][np.newaxis, :]

    # Convert cos_gwtheta to delta_gw (sin(delta) = cos(theta))
    if "cos_gwtheta" in posterior_dict:
        cos_theta = posterior_dict.pop("cos_gwtheta")
        posterior_dict["delta_gw"] = np.arcsin(cos_theta)

    # Save full posterior
    inf_data = az.from_dict(posterior=posterior_dict)
    nc_path = os.path.join(os.path.dirname(outdir), "enterprise_cw_results.nc")
    inf_data.to_netcdf(nc_path)
    print(f"Saved ArviZ NetCDF: {nc_path}")

    # Print CW parameter summary
    print("\n=== CW Parameter Summary ===")
    injections = {
        "log10_h0": -13.350, "log10_f_gw": -8.215,
        "alpha_gw": 4.067, "delta_gw": 0.140,
        "cos_iota": 0.907, "psi": 0.646, "Phi0": 0.175,
    }
    for var in cw_params:
        if var in posterior_dict:
            s = posterior_dict[var].flatten()
            q16, q50, q84 = np.percentile(s, [16, 50, 84])
            inj = injections.get(var, None)
            inj_str = f" (inj={inj:.3f})" if inj is not None else ""
            print(f"  {var:15s}: {q50:.4f} +{q84-q50:.4f} -{q50-q16:.4f}{inj_str}")

    return nc_path


def main():
    parser = argparse.ArgumentParser(description="ENTERPRISE CW search benchmark")
    parser.add_argument("--mode", choices=["fixed", "sampled"], default="fixed",
                        help="Noise mode: 'fixed' or 'sampled'")
    parser.add_argument("--n-samples", type=int, default=500_000,
                        help="Number of MCMC samples")
    parser.add_argument("--thin", type=int, default=10, help="Thinning factor")
    parser.add_argument("--n-temps", type=int, default=5, help="Number of temperature chains")
    parser.add_argument("--t-max", type=float, default=1000, help="Maximum temperature")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Skip sampling, only run post-processing")
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(
            SCRIPT_DIR, f"outputs/enterprise_cw_{args.mode}/chains/"
        )

    print(f"=== ENTERPRISE CW Search ({args.mode} noise) ===")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {args.outdir}")

    pulsars, noise_params = load_pulsars()

    print(f"\nBuilding signal model (mode={args.mode})...")
    t0 = time.time()
    pta = build_model(pulsars, noise_params, mode=args.mode)
    print(f"Model built in {time.time()-t0:.1f}s, {len(pta.params)} free parameters")

    if not args.postprocess_only:
        run_sampler(pta, args.outdir, n_samples=args.n_samples, thin=args.thin,
                    n_temps=args.n_temps, t_max=args.t_max)

    postprocess(pta, args.outdir)


if __name__ == "__main__":
    main()
