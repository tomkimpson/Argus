#!/usr/bin/env python
"""Ingest the 67 NG15 narrowband feathers into per-TOA compression caches (task 2.1).

    python workflows/ng15_nb/scripts/ingest_nb.py \
        [--discovery-dir /fred/oz022/tkimpson/external/discovery] \
        [--out-dir workflows/ng15_nb/data/ingest] \
        [--summary workflows/ng15_nb/results/ingest_summary.json]

Every feather is checked against its SHA-256 in data/ng15_nb/MANIFEST.json before use.
The caches (one .npz per pulsar) are gitignored; the summary JSON is committed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
import ingest  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--discovery-dir", default="/fred/oz022/tkimpson/external/discovery"
    )
    ap.add_argument("--out-dir", default=str(REPO / "workflows/ng15_nb/data/ingest"))
    ap.add_argument(
        "--summary", default=str(REPO / "workflows/ng15_nb/results/ingest_summary.json")
    )
    ap.add_argument("--pulsars", nargs="*", help="subset (default: all 67)")
    args = ap.parse_args()

    manifest = json.loads((REPO / "data/ng15_nb/MANIFEST.json").read_text())
    f0 = json.loads((REPO / "data/ng15_nb/f0_catalog.json").read_text())["f0_hz"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in manifest["feathers"]:
        if args.pulsars and entry["pulsar"] not in args.pulsars:
            continue
        path = Path(args.discovery_dir) / "data" / entry["file"]
        if sha256(path) != entry["sha256"]:
            sys.exit(f"{path}: SHA-256 does not match MANIFEST.json")
        cache = ingest.ingest(path, f0[entry["pulsar"]])
        ingest.save(cache, out_dir)
        rows.append(ingest.summarize(cache))
        print(json.dumps(rows[-1]), flush=True)

    lengths = [r["dmx_window_days"]["max"] for r in rows]
    totals = {
        k: int(sum(r[k] for r in rows))
        for k in (
            "n_toa",
            "n_cols",
            "n_dmx_cols",
            "n_other_cols",
            "n_toa_without_dmx",
            "n_single_freq_windows",
            "n_windows_multi_ecorr",
            "n_ecorr_epochs",
            "n_ecorr_straddling",
        )
    }
    totals["n_pulsars"] = len(rows)
    totals["max_dmx_window_days"] = float(np.max(lengths))
    totals["median_of_median_dmx_window_days"] = float(
        np.median([r["dmx_window_days"]["median"] for r in rows])
    )
    summary = {
        "manifest_commit": manifest["commit"],
        "dataset_version": manifest["dataset_version"],
        "totals": totals,
        "pulsars": rows,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
