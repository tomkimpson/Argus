#!/usr/bin/env python
"""Reduce NG15 wideband per-backend white noise to a per-pulsar Argus JSON.

This is a *data-prep* tool for the SGWB demo (TASKS.md T1.6), not part of the
Argus runtime. Real NANOGrav 15yr wideband white noise is fit **per backend**
(one EFAC + one log10 t2equad per receiver+backend combination), but Argus
applies a **single scalar EFAC/EQUAD per pulsar** with no per-backend / ECORR /
DM-noise support (see PLAN.md §2, §4). This script collapses each pulsar's
per-backend white noise into one effective ``(efac, equad)`` pair and writes
``ng15_psr_noise.json`` in the schema consumed by
``argus.utils.get_efac_equad_injections``: ``{psr: {"efac": <linear>,
"equad": <log10 seconds>}}``.

Inputs (NG15 wideband release ``noise/`` dir):
  * ``{PSR}.wb.pars.txt``  — ordered list of PTMCMC parameter *names* only.
  * ``{PSR}.wb.chain_1.txt`` — PTMCMC chain, ``N_rows x (N_params + 4)``; the
    first ``N_params`` columns are in ``pars.txt`` order, the trailing 4 are
    PTMCMC diagnostics (logpost, loglik, accept, swap).

White-noise parameter names are ``{PSR}_{flag}_efac`` (linear) and
``{PSR}_{flag}_log10_t2equad`` (log10 seconds), where ``{flag}`` is the
receiver+backend token (e.g. ``Rcvr1_2_GASP``, ``3GHz_YUPPI``). The ``{flag}``
token equals the tim ``-f`` flag value exactly, so per-backend TOA counts are
recovered from the tim files and used as collapse weights. ``dmefac``,
``log10_dmequad`` and the two ``red_noise_*`` parameters are dropped (Argus has
no DM-noise / ECORR, and red noise stays free/hierarchical for Stage 2/3).

Collapse (variance-preserving, TOA-count-weighted). Argus combines the noise as
``R = (efac*sigma)^2 + equad^2`` (``python/argus/model.py:112``), so the
effective pair is built in linear/variance space:

    efac_eff  = sum(w_b * efac_b) / sum(w_b)
    equad_eff = sqrt(sum(w_b * equad_b^2) / sum(w_b))      # equad in seconds

with ``w_b`` the per-backend TOA count (equal weights if a backend's count is
unavailable). ``equad`` is written back as ``log10(equad_eff)`` because the
consumer applies ``10 ** equad`` on read. This is an accepted demo approximation
(single scalar per pulsar); flag it in the README (T4.2).

Example
-------
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/reduce_ng15_white_noise.py --overwrite
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

# NG15 wideband release layout (verified present on disk; see PLAN.md §4).
NG15_ROOT = (
    "/fred/oz022/tkimpson/pta-solar-wind-2/data/"
    "NANOGrav15yr_PulsarTiming_v2.1.0/wideband"
)
DEFAULT_NOISE_DIR = os.path.join(NG15_ROOT, "noise")
DEFAULT_TIM_DIR = os.path.join(NG15_ROOT, "tim")

# Paths relative to the repo root (this file is workflows/ng15_sgwb_demo/scripts/).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_FEATHER_DIR = os.path.join(_REPO_ROOT, "workflows", "ng15_sgwb_demo", "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_FEATHER_DIR, "ng15_psr_noise.json")

# Sanity range for a warning only (from example_config.ini fallback priors).
EFAC_RANGE = (0.1, 3.0)
LOG10_EQUAD_RANGE = (-9.0, -5.0)

EFAC_SUFFIX = "_efac"
EQUAD_SUFFIX = "_log10_t2equad"


def pulsars_from_feathers(feather_dir):
    """Return pulsar names in the loader's order (``sorted(glob("*.feather"))``).

    Argus loads pulsars via ``sorted(glob(os.path.join(dir, "*.feather")))``
    (``data_loader.py:259``) and ``get_efac_equad_injections`` consumes the
    resulting EFAC/EQUAD arrays *positionally*. Deriving the JSON key order from
    the same sorted glob guarantees the noise JSON aligns with the residual
    matrix.
    """
    files = sorted(glob.glob(os.path.join(feather_dir, "*.feather")))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def parse_backend_params(pars_path, psr):
    """Map backend flag -> {"efac": col_idx, "equad": col_idx} from a pars file.

    ``pars.txt`` lists parameter *names* only, one per line; the line index is
    the column index into the chain. The receiver token itself contains
    underscores (``Rcvr1_2``), so the backend flag is recovered by stripping the
    ``{psr}_`` prefix and the ``_efac`` / ``_log10_t2equad`` suffix rather than
    by splitting on ``_``.
    """
    prefix = psr + "_"
    backends = collections.defaultdict(dict)
    with open(pars_path) as f:
        names = [line.strip() for line in f if line.strip()]

    for idx, name in enumerate(names):
        if not name.startswith(prefix):
            continue
        if name.endswith(EFAC_SUFFIX):
            flag = name[len(prefix) : -len(EFAC_SUFFIX)]
            backends[flag]["efac"] = idx
        elif name.endswith(EQUAD_SUFFIX):
            flag = name[len(prefix) : -len(EQUAD_SUFFIX)]
            backends[flag]["equad"] = idx
        # dmefac / log10_dmequad / red_noise_* are intentionally ignored.

    # Keep only backends with a complete efac+equad pair.
    complete = {
        flag: cols
        for flag, cols in backends.items()
        if "efac" in cols and "equad" in cols
    }
    return complete, len(names)


def count_toas_per_backend(tim_path):
    """Count TOAs per ``-f`` flag value in a tim file. Returns {flag: count}."""
    counts = collections.Counter()
    with open(tim_path) as f:
        for line in f:
            tokens = line.split()
            if "-f" in tokens:
                i = tokens.index("-f")
                if i + 1 < len(tokens):
                    counts[tokens[i + 1]] += 1
    return counts


def reduce_pulsar(psr, noise_dir, tim_dir, burn_in):
    """Collapse one pulsar's per-backend white noise to (efac_eff, log10_equad_eff)."""
    pars_path = os.path.join(noise_dir, f"{psr}.wb.pars.txt")
    chain_path = os.path.join(noise_dir, f"{psr}.wb.chain_1.txt")
    if not os.path.exists(pars_path) or not os.path.exists(chain_path):
        raise FileNotFoundError(f"missing noise files for {psr} in {noise_dir}")

    backends, n_params = parse_backend_params(pars_path, psr)
    if not backends:
        raise ValueError(f"no complete efac/equad backend pairs found for {psr}")

    # Load chain, discard burn-in, keep only the parameter columns.
    chain = np.loadtxt(chain_path)
    chain = chain[:, :n_params]
    start = int(burn_in * chain.shape[0])
    chain = chain[start:]

    tim_matches = glob.glob(os.path.join(tim_dir, f"{psr}_PINT_*.wb.tim"))
    toa_counts = count_toas_per_backend(tim_matches[0]) if tim_matches else {}
    if not tim_matches:
        print(f"  [{psr}] WARNING: no tim file found -> equal-weight collapse")

    flags = sorted(backends)
    efacs = np.array([np.median(chain[:, backends[f]["efac"]]) for f in flags])
    # log10 t2equad posterior median, then delog to seconds.
    equads = np.array([10 ** np.median(chain[:, backends[f]["equad"]]) for f in flags])
    weights = np.array([float(toa_counts.get(f, 0)) for f in flags])
    if weights.sum() == 0:
        weights = np.ones_like(weights)  # fall back to equal weights

    efac_eff = float(np.average(efacs, weights=weights))
    equad_eff = float(np.sqrt(np.average(equads**2, weights=weights)))
    log10_equad_eff = float(np.log10(equad_eff))

    detail = {
        "n_backends": len(flags),
        "n_samples": int(chain.shape[0]),
        "backends": {
            f: {"efac": float(e), "log10_equad": float(np.log10(q)), "n_toa": int(w)}
            for f, e, q, w in zip(flags, efacs, equads, weights)
        },
    }
    return efac_eff, log10_equad_eff, detail


def build_noise_json(feather_dir, noise_dir, tim_dir, burn_in):
    """Return an ordered {psr: {"efac", "equad"}} dict + per-pulsar details."""
    psrs = pulsars_from_feathers(feather_dir)
    if not psrs:
        raise FileNotFoundError(
            f"no *.feather in {feather_dir}; run ingest first (T1.3)"
        )

    noise = collections.OrderedDict()
    details = collections.OrderedDict()
    for psr in psrs:  # already in sorted-feather (== loader) order
        efac_eff, log10_equad_eff, detail = reduce_pulsar(
            psr, noise_dir, tim_dir, burn_in
        )
        noise[psr] = {"efac": efac_eff, "equad": log10_equad_eff}
        details[psr] = detail
    return noise, details


def print_summary(noise, details):
    """Print a per-pulsar table and warn on out-of-range effective values."""
    print(
        "\n{:<12} {:>4} {:>10} {:>14}".format(
            "pulsar", "nbe", "efac_eff", "log10_equad"
        )
    )
    print("-" * 44)
    for psr, params in noise.items():
        efac, log10_equad = params["efac"], params["equad"]
        nbe = details[psr]["n_backends"]
        flags = []
        if not (EFAC_RANGE[0] <= efac <= EFAC_RANGE[1]):
            flags.append("EFAC out of range")
        if not (LOG10_EQUAD_RANGE[0] <= log10_equad <= LOG10_EQUAD_RANGE[1]):
            flags.append("log10_equad out of range")
        warn = ("  <-- " + "; ".join(flags)) if flags else ""
        print(f"{psr:<12} {nbe:>4d} {efac:>10.4f} {log10_equad:>14.4f}{warn}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--noise-dir",
        default=DEFAULT_NOISE_DIR,
        help="NG15 wideband noise/ dir (pars.txt + chain_1.txt)",
    )
    parser.add_argument(
        "--tim-dir",
        default=DEFAULT_TIM_DIR,
        help="NG15 wideband tim/ dir (for per-backend TOA-count weights)",
    )
    parser.add_argument(
        "--feather-dir",
        default=DEFAULT_FEATHER_DIR,
        help="Ingested feather dir; sets the pulsar list and order",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument(
        "--burn-in",
        type=float,
        default=0.25,
        help="Fraction of each chain to discard as burn-in",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSON if it already exists",
    )
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print(f"Output exists (use --overwrite): {args.output}")
        return

    noise, details = build_noise_json(
        args.feather_dir, args.noise_dir, args.tim_dir, args.burn_in
    )
    print_summary(noise, details)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(noise, f, indent=4)
    print(f"Wrote {len(noise)} pulsars -> {args.output}")


if __name__ == "__main__":
    main()
