#!/usr/bin/env python
"""Adapter: NanoGrav *Discovery* per-pulsar feathers -> Argus feather cache + fixed noise.

NanoGrav's Discovery (github.com/nanograv/discovery) ships the full NG15 dataset as
per-pulsar ``.feather`` files, each carrying TOAs, residuals, the timing design
matrix, sky position, pulsar distance, and a *published noise dictionary*
(per-backend EFAC/EQUAD/ECORR + power-law red noise). Argus and Discovery
independently converged on the same architecture choice -- a per-pulsar feather
cache with ``enterprise`` as data-prep-only -- but use *different* on-disk schemas.

This script bridges them so Argus can run a continuous-wave (CW) search on real
NG15 data without re-running enterprise/PINT:

1. Read each Discovery feather with ``pyarrow`` only (no ``discovery`` dependency).
2. Source the scalar spin frequency F0 -- which Discovery feathers do NOT store --
   from ``scripts/ng15_f0_catalog.json`` (built by ``build_ng15_f0_catalog.py``).
3. Fold the published *per-backend* white noise into effective per-TOA errors,
   exactly matching Discovery's t2equad convention
   ``sigma_eff^2 = EFAC_b^2 * (sigma_toa^2 + EQUAD_b^2)``. This collapses NG15's
   multi-backend white noise into a form Argus's single-EFAC/EQUAD model consumes
   faithfully (we then set EFAC=1, EQUAD~0 in the emitted noise file).
4. Re-use Argus's golden ``LoadWidebandPulsarData`` path for M-scaling / P_eps and
   ``save_feather`` for the Argus-native cache.
5. Emit the two fixed-noise artifacts the Argus CW workflow expects:
   ``noise_params.json``  -> {psr: {efac, equad}} (white noise, folded => efac=1)
   ``spin_injections.pkl`` -> DataFrame[psr, optimal_sigma, optimal_gamma] (red noise)

CAVEATS (see docs/discovery_review.md):
  * ECORR (epoch-correlated white noise) is DROPPED -- Argus has no ECORR model.
    For the well-timed pulsars used in the first demo this slightly under-estimates
    the per-epoch white-noise floor.
  * Argus models red noise as a single-corner OU (integrated-frequency) process,
    whereas NG15 red noise is a power law. ``powerlaw_to_ou`` matches the residual
    PSD amplitude at the lowest sampled frequency (1/T) with a steep corner; the
    spectral *slope* is only approximate. This is the documented top item for a
    future Argus noise-model extension.

Usage
-----
    python scripts/ingest_discovery_feather.py DISCOVERY_DATA_DIR OUT_DIR \
        [--f0-catalog scripts/ng15_f0_catalog.json] \
        [--pulsars J1909-3744 J1713+0747 ...]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.feather

# Argus loader (re-used so M-scaling / P_eps / feather layout are golden-path identical).
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from argus.data_loader import LoadWidebandPulsarData  # noqa: E402

import types  # noqa: E402

FYR = 1.0 / (365.25 * 86400.0)  # 1/year in Hz (power-law reference frequency)
DEFAULT_F0_CATALOG = os.path.join(os.path.dirname(__file__), "ng15_f0_catalog.json")


# --------------------------------------------------------------------------- #
# Self-contained reader for Discovery's feather schema.
# --------------------------------------------------------------------------- #
def read_discovery_feather(path: str) -> dict:
    """Read one Discovery feather into a plain dict (pyarrow only).

    Mirrors ``discovery.pulsar.Pulsar.read_feather``: simple 1D columns, the
    flattened ``Mmat_*`` design-matrix columns, categorical ``backend_flags``,
    and the JSON ``schema.metadata`` blob (name, sky position, pdist, noisedict).
    """
    table = pyarrow.feather.read_table(path)
    names = table.column_names

    toas = table.column("toas").to_numpy()
    toaerrs = table.column("toaerrs").to_numpy()
    residuals = table.column("residuals").to_numpy()
    backend_flags = table.column("backend_flags").to_numpy().astype("U")

    mmat_cols = sorted(
        (c for c in names if re.match(r"^Mmat_\d+$", c)),
        key=lambda c: int(c.split("_")[1]),
    )
    Mmat = np.column_stack([table.column(c).to_numpy() for c in mmat_cols])

    meta = json.loads(table.schema.metadata[b"json"])
    # Discovery stores phi (=RA) and theta (=colatitude); DEC = pi/2 - theta.
    ra = float(meta["phi"])
    dec = 0.5 * np.pi - float(meta["theta"])
    pdist = meta.get("pdist", [1.0, 0.2])

    return {
        "name": meta["name"],
        "toas": toas,
        "toaerrs": toaerrs,
        "residuals": residuals,
        "Mmat": Mmat,
        "backend_flags": backend_flags,
        "RA": ra,
        "DEC": dec,
        "pdist": (float(pdist[0]), float(pdist[1])),
        "noisedict": meta.get("noisedict", {}),
    }


# --------------------------------------------------------------------------- #
# Noise handling.
# --------------------------------------------------------------------------- #
def fold_white_noise(disc: dict) -> np.ndarray:
    """Return per-TOA effective errors folding NG15 per-backend EFAC/EQUAD.

    Uses Discovery's default (t2equad) convention:
        sigma_eff(i)^2 = EFAC_b^2 * (sigma_toa(i)^2 + (10^log10_t2equad_b)^2)
    where b is the backend of TOA i. Backends without dictionary entries fall back
    to EFAC=1, EQUAD=0 (i.e. the raw TOA error).
    """
    name = disc["name"]
    nd = disc["noisedict"]
    toaerrs = disc["toaerrs"]
    flags = disc["backend_flags"]
    sigma_eff = np.array(toaerrs, dtype=float)

    for backend in sorted(set(flags)):
        mask = flags == backend
        efac = nd.get(f"{name}_{backend}_efac", 1.0)
        log10_equad = nd.get(f"{name}_{backend}_log10_t2equad", None)
        equad2 = 0.0 if log10_equad is None else 10.0 ** (2.0 * log10_equad)
        sigma_eff[mask] = np.sqrt(efac**2 * (toaerrs[mask] ** 2 + equad2))
    return sigma_eff


def powerlaw_to_ou(log10_A: float, gamma_pl: float, toas: np.ndarray, f0: float) -> tuple:
    """Approximate NG15 power-law red noise as an Argus OU spin process.

    Argus's CW filter models spin noise as an Ornstein-Uhlenbeck process on the
    spin *frequency* (corner gamma_p), whose integrated phase produces the timing
    residual r = delta_phi / f0. Its one-sided residual PSD is

        P_r(f) = 2 * sigma_p^2 / ( f0^2 * (2 pi f)^2 * (gamma_p^2 + (2 pi f)^2) ).

    The NG15 power law (enterprise convention) is

        P_pl(f) = (10^(2 log10_A)) / (12 pi^2) * fyr^(gamma_pl - 3) * f^(-gamma_pl).

    We anchor the amplitude by matching P_r = P_pl at the lowest sampled frequency
    f_ref = 1/T (where red-noise power and PTA sensitivity concentrate), placing the
    OU corner an order of magnitude below f_ref so the band sits in the steep
    (f^-4) regime. Returns (sigma_p, gamma_p). Slope fidelity is approximate -- see
    module docstring.
    """
    T = float(toas.max() - toas.min())
    f_ref = 1.0 / T
    gamma_p = 2.0 * np.pi * (0.1 * f_ref)  # OU corner, rad/s, an order below f_ref

    P_pl = (10.0 ** (2.0 * log10_A)) / (12.0 * np.pi**2) * FYR ** (gamma_pl - 3.0) * f_ref ** (-gamma_pl)
    w = 2.0 * np.pi * f_ref
    sigma_p_sq = 0.5 * P_pl * f0**2 * w**2 * (gamma_p**2 + w**2)
    sigma_p = float(np.sqrt(max(sigma_p_sq, 0.0)))
    return sigma_p, float(gamma_p)


# --------------------------------------------------------------------------- #
# Build Argus pulsar object from Discovery data.
# --------------------------------------------------------------------------- #
def to_argus_pulsar(disc: dict, sigma_eff: np.ndarray) -> LoadWidebandPulsarData:
    """Wrap Discovery data in the enterprise-attribute shim Argus's loader reads."""
    ds_psr = types.SimpleNamespace(
        toas=disc["toas"],
        toaerrs=sigma_eff,
        residuals=disc["residuals"],
        fitpars=None,
        Mmat=disc["Mmat"],
        name=disc["name"],
        _raj=disc["RA"],
        _decj=disc["DEC"],
        _pdist=disc["pdist"],
    )
    return LoadWidebandPulsarData(ds_psr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("discovery_dir", help="Directory of Discovery v1p1_*.feather files")
    ap.add_argument("out_dir", help="Output directory for Argus feathers + noise files")
    ap.add_argument("--f0-catalog", default=DEFAULT_F0_CATALOG)
    ap.add_argument(
        "--pulsars",
        nargs="*",
        default=None,
        help="Subset of pulsar names to ingest (default: all found).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.f0_catalog) as f:
        f0_catalog = json.load(f)

    disc_files = sorted(glob.glob(os.path.join(args.discovery_dir, "v1p1_*.feather")))
    if not disc_files:
        raise FileNotFoundError(f"No v1p1_*.feather files in {args.discovery_dir}")

    # Map pulsar name -> file.
    by_name = {}
    for path in disc_files:
        m = re.search(r"-([JB]\d[^/]*)\.feather$", path)
        if m:
            by_name[m.group(1)] = path

    selected = sorted(by_name) if args.pulsars is None else sorted(args.pulsars)

    noise_params = {}
    spin_rows = []
    n_ok = 0
    for name in selected:
        if name not in by_name:
            print(f"  SKIP {name}: no Discovery feather")
            continue
        if name not in f0_catalog:
            print(f"  SKIP {name}: no F0 in catalog")
            continue

        disc = read_discovery_feather(by_name[name])
        f0 = float(f0_catalog[name])
        sigma_eff = fold_white_noise(disc)

        psr = to_argus_pulsar(disc, sigma_eff)
        out_path = os.path.join(args.out_dir, f"{name}.feather")
        psr.save_feather(out_path, F0=f0)

        # White noise already folded -> identity EFAC, negligible EQUAD.
        noise_params[name] = {"efac": 1.0, "equad": -30.0}

        # Red noise -> OU approximation.
        rn_A = disc["noisedict"].get(f"{name}_red_noise_log10_A", -20.0)
        rn_g = disc["noisedict"].get(f"{name}_red_noise_gamma", 0.0)
        sigma_p, gamma_p = powerlaw_to_ou(rn_A, rn_g, disc["toas"], f0)
        spin_rows.append({"psr": name, "optimal_sigma": sigma_p, "optimal_gamma": gamma_p})

        n_ok += 1
        print(
            f"  OK  {name}: F0={f0:.3f} Hz, ntoa={len(disc['toas'])}, "
            f"sigma_eff[med]={np.median(sigma_eff):.2e}s, OU(sig={sigma_p:.2e}, gam={gamma_p:.2e})"
        )

    # Emit fixed-noise artifacts in the SAME pulsar order as sorted feathers,
    # so they align with read_multiple_feather's sorted(glob) ordering.
    with open(os.path.join(args.out_dir, "noise_params.json"), "w") as f:
        json.dump(noise_params, f, indent=2)
    spin_df = pd.DataFrame(spin_rows).sort_values("psr").reset_index(drop=True)
    spin_df.to_pickle(os.path.join(args.out_dir, "spin_injections.pkl"))

    print(f"\nIngested {n_ok} pulsars -> {args.out_dir}")
    print("  feathers, noise_params.json, spin_injections.pkl")


if __name__ == "__main__":
    main()
