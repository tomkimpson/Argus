"""Prepare the NG15 CW injection-recovery demo from *wideband* NG15 data.

Why wideband: Discovery ships NG15 as *narrowband* feathers (tens of thousands of
TOAs, hundreds of DMX columns per pulsar). Argus's CW Kalman filter is a
time-domain filter whose cost scales as n_obs x dim_M^2, so full NUTS on narrowband
data is intractable (202 GB OOM at 6 pulsars; ~11-day ETA at 5). NG15 *wideband*
data has ~40x fewer TOAs (one achromatic TOA + DM per epoch) -- exactly the scale
Argus (`LoadWidebandPulsarData`) was designed for. See ../../docs/discovery_review.md.

Steps:
1. Stage the canonical wideband par/tim pair for each requested pulsar.
2. Ingest to Argus feathers via the native `scripts/ingest_par_tim.py` (enterprise).
   This provides F0 and the design matrix directly from the timing solution -- no
   Discovery adapter or F0 catalog needed on this path.
3. Fix noise: white noise at the published TOA errors (EFAC=1, EQUAD~0), red noise
   mapped from Discovery's published per-pulsar power law to Argus's OU process
   (`scripts/ingest_discovery_feather.powerlaw_to_ou`).
4. Inject a known Earth-term CW (reusing `prepare_demo.inject_cw`).

Usage
-----
    python prepare_demo_wideband.py \
        --wb-dir /path/to/NANOGrav15yr.../wideband \
        --discovery-dir /path/to/discovery/data \
        --out-dir ./demo_data_wb \
        [--pulsars J1744-1134 B1855+09 ...] [--no-inject]
"""

import argparse
import glob
import json
import os
import re
import sys
import types

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from ingest_discovery_feather import read_discovery_feather, powerlaw_to_ou  # noqa: E402
from prepare_demo import inject_cw, DEFAULT_INJECTION  # noqa: E402
from argus.data_loader import LoadWidebandPulsarData  # noqa: E402

DEFAULT_PULSARS = ["J1944+0907", "B1855+09", "J1455-3330", "J0613-0200", "J1744-1134"]


def canonical_pair(wb_dir: str, name: str):
    """Return the canonical (no telescope suffix) wideband (par, tim) for a pulsar."""
    def pick(ext):
        cands = sorted(glob.glob(os.path.join(wb_dir, ext, f"{name}_PINT_*.wb.{ext}")))
        for c in cands:
            if re.match(rf"^{re.escape(name)}_PINT_\d+\.wb\.{ext}$", os.path.basename(c)):
                return c
        return cands[0] if cands else None
    return pick("par"), pick("tim")


def ingest_wideband(par: str, tim: str, out_path: str) -> str:
    """Read one wideband par/tim with PINT, SVD-recondition the design matrix, save.

    Two wideband-specific steps beyond the standard narrowband ingestion:
      * ``timing_package='pint'`` -- the NG15 wideband files are PINT products;
        tempo2 chokes on the ``DMDATA`` keyword.
      * SVD-orthonormalise the design matrix. Wideband keeps a large per-epoch DMX
        bank (~169 columns) but has only ~hundreds of epochs, so the raw design
        matrix is catastrophically ill-conditioned (singular-value range ~1e16) and
        Argus's ``P_eps = inv(M^T N^-1 M)`` blows up. Replacing M with its SVD left
        basis U spans the *same* timing subspace -- so the marginalised likelihood
        is unchanged -- but is well-conditioned (cond ~1e4). This mirrors
        discovery's ``makegp_timing(svd=True)``.
    """
    from enterprise.pulsar import Pulsar as EnterprisePulsar

    raw = EnterprisePulsar(par, tim, timing_package="pint")
    U, _s, _Vt = np.linalg.svd(np.asarray(raw.Mmat), full_matrices=False)
    shim = types.SimpleNamespace(
        toas=raw.toas, toaerrs=raw.toaerrs, residuals=raw.residuals,
        fitpars=list(raw.fitpars), Mmat=U, name=raw.name,
        _raj=raw._raj, _decj=raw._decj, _pdist=raw._pdist,
    )
    psr = LoadWidebandPulsarData(shim)
    f0 = LoadWidebandPulsarData.get_par_value(par, "F0")
    psr.save_feather(out_path, F0=f0)
    return raw.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wb-dir", required=True, help="NG15 wideband release dir (has par/ and tim/)")
    ap.add_argument("--discovery-dir", required=True, help="Discovery data/ dir (for red-noise values)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "demo_data_wb"))
    ap.add_argument("--pulsars", nargs="*", default=DEFAULT_PULSARS)
    ap.add_argument("--no-inject", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1+2. locate and ingest each wideband pulsar (PINT + SVD recondition).
    for name in args.pulsars:
        par, tim = canonical_pair(args.wb_dir, name)
        if not par or not tim:
            print(f"  SKIP {name}: no canonical wideband par/tim")
            continue
        out_path = os.path.join(args.out_dir, f"{name}.feather")
        ingest_wideband(par, tim, out_path)
        print(f"  ingested {name}")

    feathers = sorted(glob.glob(os.path.join(args.out_dir, "*.feather")))
    print(f"Ingested {len(feathers)} feathers")

    # 3. fixed noise: white = raw TOA errors (efac=1, equad~0); red = OU from Discovery power law
    disc_by_name = {}
    for p in glob.glob(os.path.join(args.discovery_dir, "v1p1_*.feather")):
        m = re.search(r"-([JB]\d[^/]*)\.feather$", p)
        if m:
            disc_by_name[m.group(1)] = p

    noise_params = {}
    spin_rows = []
    for fpath in feathers:
        psr = LoadWidebandPulsarData.read_feather(fpath)
        name = psr.name
        noise_params[name] = {"efac": 1.0, "equad": -30.0}
        rn_A, rn_g = -20.0, 0.0
        if name in disc_by_name:
            nd = read_discovery_feather(disc_by_name[name])["noisedict"]
            rn_A = nd.get(f"{name}_red_noise_log10_A", -20.0)
            rn_g = nd.get(f"{name}_red_noise_gamma", 0.0)
        f0 = float(getattr(psr, "F0", 1.0) or 1.0)
        sigma_p, gamma_p = powerlaw_to_ou(rn_A, rn_g, np.asarray(psr.toas), f0)
        spin_rows.append({"psr": name, "optimal_sigma": sigma_p, "optimal_gamma": gamma_p})
        print(f"  {name}: ntoa={len(psr.toas)}, dim_M={psr.M_matrix.shape[1]}, "
              f"OU(sig={sigma_p:.2e}, gam={gamma_p:.2e}), P_eps finite={np.all(np.isfinite(psr.P_eps))}")

    with open(os.path.join(args.out_dir, "noise_params.json"), "w") as f:
        json.dump(noise_params, f, indent=2)
    pd.DataFrame(spin_rows).sort_values("psr").reset_index(drop=True).to_pickle(
        os.path.join(args.out_dir, "spin_injections.pkl"))

    # 4. inject CW
    if args.no_inject:
        print("Real-data mode: no CW injected.")
        return
    print("\nInjecting Earth-term CW:")
    inject_cw(args.out_dir, DEFAULT_INJECTION)
    with open(os.path.join(args.out_dir, "injection.json"), "w") as f:
        json.dump(DEFAULT_INJECTION, f, indent=2)
    print(f"\nWideband demo data ready in {args.out_dir}")


if __name__ == "__main__":
    main()
