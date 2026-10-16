#!/usr/bin/env python
"""Compress the ingested NG15 narrowband pulsars and report the rank rule (task 2.2).

    python workflows/ng15_nb/scripts/compress_nb.py \
        [--ingest-dir workflows/ng15_nb/data/ingest] \
        [--summary workflows/ng15_nb/results/compression_summary.json] \
        [--delta-days none 7 14 30]

The summary records, per pulsar and grid width, the number of cells and scalars, the
timing columns and the directions the rank rule keeps, the column dominating each dropped
direction, and the smallest squared singular-value ratio (the rank rule's margin).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
import compress  # noqa: E402
import ingest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", default=str(REPO / "workflows/ng15_nb/data/ingest"))
    ap.add_argument(
        "--summary",
        default=str(REPO / "workflows/ng15_nb/results/compression_summary.json"),
    )
    ap.add_argument("--delta-days", nargs="*", default=["none", "7", "14", "30"])
    args = ap.parse_args()
    deltas = [None if d == "none" else float(d) for d in args.delta_days]

    rows = []
    for path in sorted(Path(args.ingest_dir).glob("*.npz")):
        cache = ingest.load(path)
        row = {"pulsar": cache["name"], "n_toa": int(cache["toas"].size)}
        for d in deltas:
            comp = compress.compress_pulsar(cache, delta_days=d)
            row["dmx_window" if d is None else f"{d:g}d"] = {
                "n_cells": int(comp["times"].size),
                "n_scalars": int(comp["has_scalar"].sum()),
                "n_timing_cols": int(comp["basis"].shape[0]),
                "n_kept_directions": int(comp["basis"].shape[1]),
                "dropped": comp["dropped"].tolist(),
                "min_singular_ratio2": float(comp["singular_ratio2"].min()),
            }
        rows.append(row)
        print(json.dumps(row), flush=True)

    keys = ["dmx_window" if d is None else f"{d:g}d" for d in deltas]
    totals = {
        k: {
            "n_cells": int(sum(r[k]["n_cells"] for r in rows)),
            "n_scalars": int(sum(r[k]["n_scalars"] for r in rows)),
            "n_timing_cols": int(sum(r[k]["n_timing_cols"] for r in rows)),
            "n_kept_directions": int(sum(r[k]["n_kept_directions"] for r in rows)),
            "n_dropped": int(sum(len(r[k]["dropped"]) for r in rows)),
        }
        for k in keys
    }
    out = {"rank_tol": 1e-14, "totals": totals, "pulsars": rows}
    Path(args.summary).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
