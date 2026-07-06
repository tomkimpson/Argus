"""Fast sanity check for the NG15 CW injection demo (no NUTS required).

Loads the prepared ``demo_data`` (with an injected Earth-term CW), evaluates Argus's
CW likelihood on a grid of GW frequencies holding the other parameters at their
injected values, and confirms the likelihood:

  * peaks at the injected f_gw, and
  * is substantially higher with the CW than without (detection).

This exercises the whole adapter -> load -> CW Kalman filter path quickly. Full
parameter recovery is the job of the NUTS run (``run_analysis.py``), which is best
run on an A100 (the narrowband NG15 TOAs are heavy).

Usage
-----
    python verify_injection.py [--data-dir ./demo_data] [--npoints 13]
"""

import argparse
import configparser
import json
import os
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

from argus import io_manager
from argus.data_loader import LoadWidebandPulsarData
from argus.cw_kalman_filter import CWKalmanFilter
from argus.bayesian_inference import CWParameters
from argus.gravitational_waves import antenna_pattern_single


def _init_logger():
    cfg = configparser.ConfigParser()
    cfg.add_section("Logging")
    cfg.set("Logging", "level", "WARNING")
    cfg.set("Logging", "enable_file_logging", "false")
    io_manager.setup_single_logger(cfg, enable_file_logging=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "demo_data"))
    ap.add_argument("--npoints", type=int, default=13)
    args = ap.parse_args()

    _init_logger()
    with open(os.path.join(args.data_dir, "injection.json")) as f:
        inj = json.load(f)

    data = LoadWidebandPulsarData.get_processed_residuals(args.data_dir, mode="cw")
    md = data["metadata"]
    names = list(md["name"])
    Npsr = len(names)
    print(f"Loaded {Npsr} pulsars: {names}")

    KF = CWKalmanFilter(data=data, include_pulsar_term=False, phase_parameterization=True)

    np_json = json.load(open(os.path.join(args.data_dir, "noise_params.json")))
    spin = pd.read_pickle(os.path.join(args.data_dir, "spin_injections.pkl")).set_index("psr")
    EFAC = jnp.array([np_json[n]["efac"] for n in names])
    EQUAD = jnp.array([10.0 ** np_json[n]["equad"] for n in names])
    sigma_p = jnp.array([spin.loc[n, "optimal_sigma"] for n in names])
    gamma_p = jnp.array([spin.loc[n, "optimal_gamma"] for n in names])

    def theta_at(f_gw, h0):
        return CWParameters(
            alpha_gw=inj["alpha_gw"], delta_gw=inj["delta_gw"], f_gw=f_gw, h0=h0,
            cos_iota=inj["cos_iota"], psi=inj["psi"], Phi0=inj["Phi0"],
            chi=jnp.zeros(Npsr), gamma_p=gamma_p, sigma_p=sigma_p, EFAC=EFAC, EQUAD=EQUAD)

    f_inj = inj["f_gw"]
    grid = np.geomspace(0.5 * f_inj, 2.0 * f_inj, args.npoints)
    grid = np.unique(np.concatenate([grid, [f_inj]]))  # ensure injection is sampled
    ll = np.array([float(KF.get_likelihood(theta_at(f, inj["h0"]))) for f in grid])

    peak = grid[np.argmax(ll)]
    ll_inj = float(KF.get_likelihood(theta_at(f_inj, inj["h0"])))
    ll_nocw = float(KF.get_likelihood(theta_at(f_inj, 1e-20)))

    print(f"\nInjected f_gw = {f_inj:.3e} Hz, h0 = {inj['h0']:.1e}")
    print(f"Scan peak     = {peak:.3e} Hz  (over {len(grid)} points)")
    print(f"Detection:  logL(injected) - logL(no CW) = {ll_inj - ll_nocw:+.1f}")
    print(f"Band depth: logL(max) - logL(min)        = {ll.max() - ll.min():.1f}")

    ok_peak = abs(np.log10(peak) - np.log10(f_inj)) < 1e-6
    ok_det = (ll_inj - ll_nocw) > 5.0
    if ok_peak and ok_det:
        print("\nPASS: CW detected and likelihood peaks at the injected frequency.")
    else:
        print(f"\nCHECK: peak_ok={ok_peak}, detection_ok={ok_det}")
        sys.exit(1)


if __name__ == "__main__":
    main()
