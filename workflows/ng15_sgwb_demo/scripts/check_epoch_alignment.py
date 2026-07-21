#!/usr/bin/env python
"""Diagnose epoch alignment across the NG15 wideband pulsar subset (RISK A gate).

Argus's GWB (Hellings-Downs) path has *no missing-observation mechanism*. Its
joint Kalman filter reaches the residuals through
``LoadWidebandPulsarData.process_pulsar_residuals_by_epoch``
(``python/argus/data_loader.py:99``), which aligns pulsars **purely by row
index** -- row ``k`` is assumed to be the same observation epoch in every pulsar
-- and raises ``ValueError`` unless every pulsar has an *identical* number of
TOAs. There is no time-matching or interpolation. The MDC2 data this path was
validated on is idealized (33 pulsars x 185 synchronous TOAs). Real NG15 pulsars
are ragged (differing TOA counts, unaligned epochs), so the GWB path cannot
consume them as ingested -- this is **RISK A, the make-or-break gate** for the
whole SGWB milestone (PLAN sec 2).

This script does *not* fix alignment (that is T1.5). It answers, before any
binning code or GPU time is spent: *can these pulsars be placed on a common
epoch grid, and at what cadence?* It:

1. Loads every ``data/*.feather`` via ``read_multiple_feather`` -- which returns
   per-pulsar DataFrames **without** triggering the shape check -- and converts
   the ``toas`` column (seconds) to MJD.
2. Reports per-pulsar TOA/epoch counts, MJD span, baseline, and the
   epoch-spacing distribution (exposes the true observing cadence).
3. Reports the common observing window shared by all pulsars.
4. Sweeps candidate binning cadences and, for each, computes the load-bearing
   metric: the number of grid cells occupied by *all* pulsars (the "joint
   epochs" that survive positional alignment), the per-pulsar retained-data
   fraction, and the collision rate (TOAs per occupied bin).
5. Prints a single FEASIBLE / INFEASIBLE verdict plus a recommended cadence. If
   even the best cadence yields too few joint epochs for inference, it prints an
   explicit STOP-and-flag-for-review message (PLAN sec 7 RISK A escalation).

CPU only, read-only. No library edits.

Example
-------
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/check_epoch_alignment.py
"""

import argparse
import glob
import os
import sys

import numpy as np

# Add the repo's python/ directory to sys.path so ``argus`` imports standalone,
# mirroring workflows/ng15_sgwb_demo/run_analysis.py. This file sits one level
# deeper (scripts/), so walk up four dirnames to reach the repo root.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(_PROJECT_ROOT, "python"))

from argus.data_loader import LoadWidebandPulsarData  # noqa: E402

SEC_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25

# Candidate common-grid cadences to evaluate (days). NG15 observes roughly
# monthly, so ~30 d is the starting guess (PLAN sec 6); finer/coarser bracket it.
CANDIDATE_CADENCES_DAYS = [7, 14, 30, 60]

# Below this many joint epochs, a joint GWB fit is too data-starved to be worth
# pursuing on the subset -- escalate rather than proceed silently.
MIN_VIABLE_JOINT_EPOCHS = 50

# Default feather directory: workflows/ng15_sgwb_demo/data (two levels up + data).
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def load_epochs(data_dir):
    """Load per-pulsar TOA epochs (MJD) from every feather in ``data_dir``.

    Uses :meth:`LoadWidebandPulsarData.read_multiple_feather`, which builds the
    per-pulsar DataFrames *without* invoking the epoch-alignment shape check, so
    ragged data loads cleanly here.

    Parameters
    ----------
    data_dir : str
        Directory holding ``*.feather`` caches (the ``staging_subset/`` dir and
        ``.gitkeep`` are skipped naturally by the ``*.feather`` glob).

    Returns
    -------
    list of tuple of (str, numpy.ndarray)
        ``(name, mjd)`` per pulsar, where ``mjd`` is the raw (unsorted) TOA
        array in MJD.

    Raises
    ------
    FileNotFoundError
        If no ``*.feather`` files are found in ``data_dir``.
    """
    feather_files = sorted(glob.glob(os.path.join(data_dir, "*.feather")))
    if not feather_files:
        raise FileNotFoundError(f"No *.feather files found in {data_dir}")

    pulsar_dfs, metadata, _, _ = LoadWidebandPulsarData.read_multiple_feather(
        feather_files
    )

    epochs = []
    for name, df in zip(metadata["name"].tolist(), pulsar_dfs):
        mjd = df["toas"].to_numpy(dtype=float) / SEC_PER_DAY
        epochs.append((name, mjd))
    return epochs


def report_per_pulsar(epochs):
    """Print a per-pulsar table of TOA counts, span, and epoch spacing."""
    print("=" * 92)
    print("PER-PULSAR EPOCH SUMMARY")
    print("=" * 92)
    header = (
        f"{'pulsar':<12} {'#TOA':>6} {'#epoch':>7} {'MJD_min':>9} {'MJD_max':>9} "
        f"{'yrs':>6} {'d_p05':>7} {'d_med':>7} {'d_p95':>7}"
    )
    print(header)
    print("-" * 92)
    for name, mjd in epochs:
        uniq = np.unique(mjd)
        spacings = np.diff(uniq) if uniq.size > 1 else np.array([np.nan])
        baseline_yr = (mjd.max() - mjd.min()) / DAYS_PER_YEAR
        print(
            f"{name:<12} {mjd.size:>6d} {uniq.size:>7d} "
            f"{mjd.min():>9.1f} {mjd.max():>9.1f} {baseline_yr:>6.1f} "
            f"{np.nanpercentile(spacings, 5):>7.1f} "
            f"{np.nanmedian(spacings):>7.1f} "
            f"{np.nanpercentile(spacings, 95):>7.1f}"
        )
    print(
        "\nColumns: #epoch = distinct MJD values; d_p05/d_med/d_p95 = "
        "5th/50th/95th percentile of gaps between distinct epochs (days).\n"
    )


def report_overlap(epochs):
    """Print the common observing window shared by all pulsars.

    Returns
    -------
    tuple of float
        ``(global_min, global_max, common_lo, common_hi)`` in MJD, where the
        common window is ``[max(per-pulsar mins), min(per-pulsar maxes)]``.
    """
    mins = np.array([mjd.min() for _, mjd in epochs])
    maxes = np.array([mjd.max() for _, mjd in epochs])
    global_min, global_max = mins.min(), maxes.max()
    common_lo, common_hi = mins.max(), maxes.min()

    print("=" * 92)
    print("CROSS-PULSAR OVERLAP")
    print("=" * 92)
    print(
        f"Full union window : MJD {global_min:.1f} -> {global_max:.1f} "
        f"({(global_max - global_min) / DAYS_PER_YEAR:.1f} yr)"
    )
    if common_hi > common_lo:
        print(
            f"Common window     : MJD {common_lo:.1f} -> {common_hi:.1f} "
            f"({(common_hi - common_lo) / DAYS_PER_YEAR:.1f} yr, "
            "shared by all pulsars)"
        )
    else:
        print("Common window     : NONE -- pulsar baselines do not all overlap!")
    print()
    return global_min, global_max, common_lo, common_hi


def simulate_binning(epochs, cadence_days, grid_start, grid_end):
    """Bin each pulsar onto a common grid and measure joint-epoch coverage.

    A "joint epoch" is a grid cell occupied by *every* pulsar -- the only rows
    that survive ``process_pulsar_residuals_by_epoch``'s positional alignment
    after binning (T1.5 will average the TOAs within each such cell).

    Parameters
    ----------
    epochs : list of (str, numpy.ndarray)
        Per-pulsar ``(name, mjd)`` as returned by :func:`load_epochs`.
    cadence_days : float
        Grid cell width in days.
    grid_start, grid_end : float
        Grid extent in MJD (the full union window).

    Returns
    -------
    dict
        Metrics: ``cadence``, ``n_joint`` (joint epochs), ``retained_min`` /
        ``retained_med`` (per-pulsar fraction of TOAs landing in joint cells),
        ``mean_toa_per_occ`` (collision rate across occupied cells), and
        ``toas_dropped`` (total TOAs outside joint cells).
    """
    n_bins = int(np.ceil((grid_end - grid_start) / cadence_days)) + 1

    occupied_sets = []
    bin_indices = []
    for _, mjd in epochs:
        idx = np.floor((mjd - grid_start) / cadence_days).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        bin_indices.append(idx)
        occupied_sets.append(set(idx.tolist()))

    joint_bins = set.intersection(*occupied_sets) if occupied_sets else set()
    n_joint = len(joint_bins)

    retained = []
    toas_dropped = 0
    occ_counts = []  # TOAs per occupied (pulsar, bin) cell, for collision rate
    for idx in bin_indices:
        in_joint = np.isin(idx, list(joint_bins))
        retained.append(in_joint.mean() if idx.size else 0.0)
        toas_dropped += int((~in_joint).sum())
        # occupancy of each occupied bin for this pulsar
        _, counts = np.unique(idx, return_counts=True)
        occ_counts.extend(counts.tolist())

    return {
        "cadence": cadence_days,
        "n_joint": n_joint,
        "retained_min": float(np.min(retained)) if retained else 0.0,
        "retained_med": float(np.median(retained)) if retained else 0.0,
        "mean_toa_per_occ": float(np.mean(occ_counts)) if occ_counts else 0.0,
        "toas_dropped": toas_dropped,
    }


def report_sweep(epochs, grid_start, grid_end, cadences):
    """Run the cadence sweep and print the feasibility table.

    Returns
    -------
    list of dict
        One :func:`simulate_binning` result per cadence.
    """
    total_toas = sum(mjd.size for _, mjd in epochs)
    print("=" * 92)
    print("BINNING-CADENCE SWEEP")
    print("=" * 92)
    print(
        f"{'cadence(d)':>10} {'joint_epochs':>13} {'retain_min':>11} "
        f"{'retain_med':>11} {'toa/bin':>8} {'toa_dropped':>12}"
    )
    print("-" * 92)
    results = []
    for cad in cadences:
        r = simulate_binning(epochs, cad, grid_start, grid_end)
        results.append(r)
        print(
            f"{r['cadence']:>10.0f} {r['n_joint']:>13d} "
            f"{r['retained_min']:>11.2f} {r['retained_med']:>11.2f} "
            f"{r['mean_toa_per_occ']:>8.2f} "
            f"{r['toas_dropped']:>7d}/{total_toas:<d}"
        )
    print(
        "\njoint_epochs = grid cells occupied by ALL pulsars (survive alignment).\n"
        "retain_min/med = per-pulsar fraction of TOAs kept (min & median).\n"
        "toa/bin = mean TOAs per occupied cell (>1 => averaging/collisions).\n"
        "toa_dropped = TOAs outside joint cells (alignment cost; NOT silent).\n"
    )
    return results


def verdict(results):
    """Print the feasibility verdict and recommended cadence.

    Picks the cadence maximising joint epochs (finer cadence breaks ties, to
    preserve time resolution). Escalates if the best is below
    :data:`MIN_VIABLE_JOINT_EPOCHS`.
    """
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)

    # Max joint epochs; on a tie prefer the finer (smaller) cadence.
    best = max(results, key=lambda r: (r["n_joint"], -r["cadence"]))

    if best["n_joint"] >= MIN_VIABLE_JOINT_EPOCHS:
        print(
            f"FEASIBLE: a common epoch grid is viable. Recommended cadence = "
            f"{best['cadence']:.0f} d, giving {best['n_joint']} joint epochs "
            f"(per-pulsar retention >= {best['retained_min']:.0%})."
        )
        print(
            f"\nNext (T1.5): build_aligned_feathers.py should bin onto a "
            f"{best['cadence']:.0f}-day grid, keep only the {best['n_joint']} "
            f"joint epochs, and average TOAs/residuals/errors within each cell. "
            f"That sets the joint filter's nepoch = {best['n_joint']}."
        )
    else:
        print(
            f"INFEASIBLE at the tried cadences: best is {best['n_joint']} joint "
            f"epochs at {best['cadence']:.0f} d, below the "
            f"{MIN_VIABLE_JOINT_EPOCHS}-epoch viability floor."
        )
        print(
            "\n*** STOP -- FLAG FOR REVIEW (RISK A). *** The subset cannot be "
            "aligned onto a shared grid with enough joint epochs for a joint GWB "
            "fit. Do NOT proceed to T1.5 blindly. Options to discuss: widen the "
            "cadence sweep, drop the sparsest pulsar, or extend "
            "process_pulsar_residuals_by_epoch with a masked-epoch mechanism "
            "(last resort, library change -- PLAN sec 7)."
        )
    print()


def run(data_dir, cadences):
    """Load feathers, print all reports, and return the sweep results."""
    print(f"Reading feathers from: {data_dir}\n")
    epochs = load_epochs(data_dir)
    print(f"Loaded {len(epochs)} pulsars.\n")

    report_per_pulsar(epochs)
    global_min, global_max, common_lo, common_hi = report_overlap(epochs)
    if common_hi <= common_lo:
        print(
            "*** STOP -- FLAG FOR REVIEW (RISK A): no common baseline overlap; "
            "joint alignment is impossible for this subset. ***\n"
        )
        return []

    results = report_sweep(epochs, global_min, global_max, cadences)
    verdict(results)
    return results


def main():
    """Parse command-line arguments and run the epoch-alignment diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory of ingested *.feather files (default: the workflow data/ dir)",
    )
    parser.add_argument(
        "--cadences",
        nargs="*",
        type=float,
        default=CANDIDATE_CADENCES_DAYS,
        help="Candidate binning cadences in days to evaluate",
    )
    args = parser.parse_args()
    run(args.data_dir, args.cadences)


if __name__ == "__main__":
    main()
