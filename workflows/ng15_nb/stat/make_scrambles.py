#!/usr/bin/env python
"""Seeded sky-scramble ensemble for the 67 NG15 positions (detection-statistic task 1.3).

    python workflows/ng15_nb/stat/make_scrambles.py [--n 5000] [--seed 20261120]

Uses `ng15_sgwb_demo/scripts/sky_scrambles.generate_scrambles` unchanged (spec D4), with
the spec's match threshold 0.17 passed explicitly (the module default is 0.2). The true
ORF is the one the filter uses, `union.hd_matrix` over the ingest caches' positions
(unit diagonal), so Γ − I has a zero diagonal for every ORF scored.

Writes the archive to the gitignored `workflows/ng15_nb/data/scrambles/` and a tracked
summary (seed, count, threshold, matches, acceptance, sha-256) to
`results/stat/opening/scrambles_summary.json`. Then regenerates the first 10 from the
seed and checks they are bit-identical.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orf_gradient as og  # noqa: E402

THRESHOLD = 0.17
DATA = og.DEFAULT_INGEST.parent / "scrambles"  # beside the ingest caches (main checkout)
SUMMARY = og.WF / "results/stat/opening/scrambles_summary.json"


def true_orf():
    import common
    import ingest
    import union

    caches = [ingest.load(og.DEFAULT_INGEST / f"{n}.npz") for n in common.pulsars()]
    return union.hd_matrix(caches)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    sky = og.load_script("sky_scrambles")
    hd = true_orf()
    t = time.perf_counter()
    ens = sky.generate_scrambles(hd, args.n, seed=args.seed, match_threshold=THRESHOLD)
    wall = time.perf_counter() - t
    orfs = ens["orfs"]
    matches = np.array([sky.orf_match(o, hd) for o in orfs])
    assert len(orfs) == args.n
    assert matches.max() < THRESHOLD, matches.max()
    assert np.array_equal(matches, ens["matches"])
    assert np.all(np.diagonal(orfs, axis1=1, axis2=2) == 1.0)
    assert np.all(np.diag(hd) == 1.0)

    again = sky.generate_scrambles(hd, 10, seed=args.seed, match_threshold=THRESHOLD)
    regenerated = bool(np.array_equal(again["orfs"], orfs[:10]))
    assert regenerated

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"ng15nb_stat_ens{args.n}_seed{args.seed}.npz"
    np.savez_compressed(out, orfs=orfs, matches=matches, true_orf=hd, seed=args.seed)
    summary = {
        "spec": "spec/detection-statistic.md task 1.3",
        "generator": "workflows/ng15_sgwb_demo/scripts/sky_scrambles.py::generate_scrambles",
        "true_orf": "union.hd_matrix over the 67 ingest caches (manifest order)",
        "path": str(out),
        "sha256": og.sha256(out),
        "seed": args.seed,
        "n_scrambles": int(len(orfs)),
        "match_threshold": THRESHOLD,
        "match_max": float(matches.max()),
        "match_mean": float(matches.mean()),
        "n_attempts": int(ens["n_attempts"]),
        "acceptance_rate": float(ens["acceptance_rate"]),
        "wall_s": wall,
        "first_10_regenerated_bit_identical": regenerated,
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
