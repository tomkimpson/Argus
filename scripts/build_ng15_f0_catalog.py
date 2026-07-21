#!/usr/bin/env python
"""Build a JSON catalog of NG15 pulsar spin frequencies (F0).

Discovery's bundled NG15 ``.feather`` files do not carry the absolute spin
frequency F0 (only a fit *design matrix*), but Argus's CW Kalman filter needs it:
the observation vector maps the spin-phase state to a timing residual via
``h[0] = 1 / F0`` (``cw_kalman_filter._build_per_pulsar_H_vectors``). F0 is the one
scalar we must source from the official NG15 timing release.

This one-time script scans the NG15 narrowband ``par`` directory, extracts the F0
value from each canonical ``{PSR}_PINT_*.nb.par`` file, and writes a small JSON
``{pulsar_name: F0_hz}`` catalog into the repo so the Discovery->Argus adapter
(``scripts/ingest_discovery_feather.py``) stays self-contained and does not depend
on an external par directory at run time.

Usage
-----
    python scripts/build_ng15_f0_catalog.py [NG15_PAR_DIR] [-o OUT_JSON]

Defaults match the copy of the NG15 release already on the OzSTAR cluster.
"""

import argparse
import glob
import json
import os
import re

DEFAULT_PAR_DIR = (
    "/fred/oz022/tkimpson/pta-solar-wind-2/data/"
    "NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par"
)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "ng15_f0_catalog.json")


def get_par_value(filename: str, parameter: str) -> float | None:
    """Return the numeric value of ``parameter`` from a tempo2/PINT par file."""
    with open(filename) as f:
        for line in f:
            fields = line.split()
            if fields and fields[0] == parameter:
                try:
                    return float(fields[1])
                except (IndexError, ValueError):
                    return None
    return None


def canonical_par(par_dir: str, name: str) -> str | None:
    """Find the canonical ``{name}_PINT_*.nb.par`` file for a pulsar.

    Excludes telescope-specific variants (e.g. ``B1937+21ao_...``) by requiring
    that the token before ``_PINT`` is exactly the pulsar name.
    """
    candidates = sorted(glob.glob(os.path.join(par_dir, f"{name}_PINT_*.nb.par")))
    for cand in candidates:
        base = os.path.basename(cand)
        if re.match(rf"^{re.escape(name)}_PINT_\d+\.nb\.par$", base):
            return cand
    return candidates[0] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("par_dir", nargs="?", default=DEFAULT_PAR_DIR)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    # Discover pulsar names directly from the canonical par files present.
    names = set()
    for path in glob.glob(os.path.join(args.par_dir, "*_PINT_*.nb.par")):
        m = re.match(r"^([JB]\d[^_]*)_PINT_\d+\.nb\.par$", os.path.basename(path))
        if m:
            names.add(m.group(1))

    catalog: dict[str, float] = {}
    missing = []
    for name in sorted(names):
        par = canonical_par(args.par_dir, name)
        f0 = get_par_value(par, "F0") if par else None
        if f0 is None:
            missing.append(name)
            continue
        catalog[name] = f0

    with open(args.out, "w") as f:
        json.dump(catalog, f, indent=2, sort_keys=True)

    print(f"Wrote {len(catalog)} F0 values to {args.out}")
    if missing:
        print(f"WARNING: no F0 found for: {missing}")


if __name__ == "__main__":
    main()
