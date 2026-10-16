"""Spec N2: the common-process amplitude under each Discovery kernel, at f = 1/T and 1/(5 yr).

Spec task 1.6. Each run's posterior draws are pushed through its own one-sided residual
PSD (s²/Hz): Discovery's ``powerlaw`` for the PL runs, ``ou_psd.gw_psd_onesided`` for the
OU runs. The result is a like-for-like amplitude across kernels, which a log10 A and an h_a
are not. T is the array span Discovery uses (``getspan``: first to last TOA over the 67
feathers pinned in data/ng15_nb/MANIFEST.json).

    conda activate Argus
    python workflows/ng15_nb/reference/kernel_systematic.py [--discovery-dir DIR] \\
        --out .../readouts/n2_psd.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow.feather as pf

from ou_psd import gw_psd_onesided

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = REPO_ROOT / "workflows/ng15_nb/outputs"
DEFAULT_DISCOVERY_DIR = os.environ.get(
    "NG15_DISCOVERY_DIR", "/fred/oz022/tkimpson/external/discovery"
)
YR_S = 365.25 * 86400.0
FYR = 1.0 / YR_S  # discovery.const.fyr
RUNS = [
    f"ng15nb_{k}_eps{e}" for k in ("pl", "ou14", "ou30") for e in ("000", "100")
] + ["ng15nb_pl13_eps100"]


def array_span_s(discovery_dir):
    entries = json.loads((REPO_ROOT / "data/ng15_nb/MANIFEST.json").read_text())[
        "feathers"
    ]
    lo, hi = math.inf, -math.inf
    for e in entries:
        t = (
            pf.read_table(Path(discovery_dir) / "data" / e["file"], columns=["toas"])
            .column("toas")
            .to_numpy()
        )
        lo, hi = min(lo, t.min()), max(hi, t.max())
    if len(entries) != 67 or not hi > lo:
        raise SystemExit(
            f"array span from {len(entries)} manifest feathers is {hi - lo}; expected 67 and > 0"
        )
    return hi - lo


def pl_psd_onesided(f, log10_a, gamma):
    """discovery.signals.powerlaw without the df factor."""
    return (
        10.0 ** (2.0 * log10_a)
        / 12.0
        / math.pi**2
        * FYR ** (gamma - 3.0)
        * f ** (-gamma)
    )


def log10_psd_draws(run, f):
    import arviz as az

    post = az.from_netcdf(OUTPUTS / run / "samples.nc").posterior
    if "gw_log10_A" in post.data_vars:
        a = post["gw_log10_A"].values.ravel()
        g = (
            post["gw_gamma"].values.ravel()
            if "gw_gamma" in post.data_vars
            else np.full_like(a, 13.0 / 3.0)
        )
        return np.log10(pl_psd_onesided(f, a, g))
    p, g = (post[k].values.ravel() for k in ("gw_log10_pivot_psd", "gw_log10_gamma_a"))
    return np.log10(gw_psd_onesided(f, p, g))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--discovery-dir",
        default=DEFAULT_DISCOVERY_DIR,
        help="Discovery clone (default $NG15_DISCOVERY_DIR)",
    )
    ap.add_argument("--out")
    args = ap.parse_args()
    span = array_span_s(args.discovery_dir)
    freqs = {"1/T": 1.0 / span, "1/(5 yr)": 1.0 / (5.0 * YR_S)}
    res = {
        "T_yr": span / YR_S,
        "freqs_hz": freqs,
        "units": "log10 one-sided residual PSD, s^2/Hz",
        "runs": {},
    }
    for run in RUNS:
        row = {}
        for name, f in freqs.items():
            q16, q50, q84 = np.percentile(log10_psd_draws(run, f), [16, 50, 84])
            row[name] = {
                "median": q50,
                "lo68": q16,
                "hi68": q84,
                "sigma": 0.5 * (q84 - q16),
            }
        res["runs"][run] = row
    # C3's amplitude check: A at fixed γ = 13/3. NANOGrav quote a median and a 90% interval.
    import arviz as az

    a = (
        10.0
        ** az.from_netcdf(OUTPUTS / "ng15nb_pl13_eps100/samples.nc")
        .posterior["gw_log10_A"]
        .values.ravel()
    )
    res["A_HD_13_3"] = dict(
        zip(("p05", "p16", "p50", "p84", "p95"), np.percentile(a, [5, 16, 50, 84, 95]))
    )
    print(
        "A_HD,13/3 percentiles 5/16/50/84/95: "
        + "  ".join(f"{v:.3e}" for v in res["A_HD_13_3"].values())
    )
    print(f"T = {res['T_yr']:.3f} yr; {res['units']}; median [16, 84]")
    for run, row in res["runs"].items():
        cells = "   ".join(
            f"{n}: {v['median']:.3f} [{v['lo68']:.3f}, {v['hi68']:.3f}]"
            for n, v in row.items()
        )
        print(f"  {run:22s} {cells}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
