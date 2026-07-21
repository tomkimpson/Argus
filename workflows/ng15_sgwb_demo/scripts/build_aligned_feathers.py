#!/usr/bin/env python
"""Build epoch-aligned NG15 wideband feathers for Argus's GWB path (RISK A resolver).

Argus's GWB (Hellings-Downs) path aligns pulsars **purely by row index** in
``LoadWidebandPulsarData.process_pulsar_residuals_by_epoch``
(``python/argus/data_loader.py:99``) and raises ``ValueError`` unless every pulsar
has an identical number of TOAs (row ``k`` = the same epoch for all pulsars). Real
NG15 pulsars are ragged (364-1493 TOAs), so the ingested ``data/*.feather`` cannot
be consumed as-is. The T1.4 diagnostic (``check_epoch_alignment.py``) established
that a **30-day grid over the union window yields 78 "joint epochs"** (grid cells
occupied by *every* pulsar) -- the resolution this script realizes.

For each pulsar it:

1. Loads the ragged feather (raw design matrix + fit-parameter names).
2. Drops per-epoch ``DMX_*`` dispersion-measure nuisance columns. This is a
   *deliberate, necessary* simplification (PLAN sec 2/4: DM noise is not modeled).
   The full binned design matrix has rank == the epoch count (78) for every
   pulsar, so without this drop the timing model would span all of R^78 and
   **absorb the entire GW signal**. Dropping ``DMX_*`` leaves ~9-26 astrophysical
   timing parameters (spin, astrometry, binary, JUMPs), so ``P_eps`` is finite and
   full rank and genuine residual signal survives the timing marginalization.
3. Bins TOAs onto the common 30-day grid with **inverse-variance epoch averaging**
   (standard PTA averaging): for TOAs ``i`` in a joint bin with weights
   ``w_i = (1/sig_i^2) / sum_j(1/sig_j^2)``, the binned residual, TOA and design
   row are all ``sum_i w_i * x_i`` (identical weights preserve the linear timing
   model ``r = M beta + n``) and the combined error is ``1/sqrt(sum_i 1/sig_i^2)``.
   Rows are emitted in ``sorted(joint_bins)`` order so row ``k`` = the same epoch
   for every pulsar.
4. Re-applies the all-zero-column drop (mirrors ``drop_degenerate_columns`` in
   ``scripts/ingest_par_tim.py:27``) to columns that lost all support in the joint
   bins, then re-writes an aligned feather via the stock ``save_feather`` schema.

The likelihood equivalence used above (the design matrix enters only through
``M_scaled P_eps M_scaled^T = M (M^T N^-1 M)^-1 M^T``, invariant under any full-rank
column reparametrization) means the optional SVD conditioning fallback is exact --
but it is a safety net only; the semantic ``DMX_*`` drop, not re-conditioning, is
what makes the fit well-posed.

Aligned feathers are written to a ``data/aligned/`` subdir (kept separate from the
ragged originals; the stock loader's ``*.feather`` glob is non-recursive). The run
ends by calling ``get_processed_residuals(mode="gwb")`` on the output and asserting
it returns clean ``(nepoch, Npsr)`` matrices -- exactly T1.5's done-criterion.

CPU only. No library edits.

Example
-------
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/build_aligned_feathers.py --overwrite
"""

import argparse
import glob
import os
import sys
import types

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

# The T1.4 diagnostic recommended a 30-day grid -> 78 joint epochs for this subset.
DEFAULT_CADENCE_DAYS = 30.0
EXPECTED_JOINT_EPOCHS = 78

# Below this many joint epochs the joint GWB fit is too data-starved to pursue --
# escalate rather than proceed silently (matches check_epoch_alignment.py).
MIN_VIABLE_JOINT_EPOCHS = 50

# Condition number of M^T N^-1 M above which the timing marginalization is flagged
# as numerically stiff (informational; the stock loader still tolerates it).
COND_WARN_THRESHOLD = 1e12

# Default I/O: workflows/ng15_sgwb_demo/data and its aligned/ subdir.
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def load_pulsars(data_dir):
    """Load every ragged ``*.feather`` in ``data_dir`` as full pulsar objects.

    Uses :meth:`LoadWidebandPulsarData.read_feather` per file (not
    ``read_multiple_feather``) so the raw, unscaled ``M_matrix`` and ``fitpars`` are
    available for column reduction -- ``read_multiple_feather`` only exposes the
    already-scaled design matrices.

    Parameters
    ----------
    data_dir : str
        Directory of ingested ``*.feather`` files (its ``aligned/`` subdir and
        ``.gitkeep`` are skipped naturally by the non-recursive ``*.feather`` glob).

    Returns
    -------
    list of LoadWidebandPulsarData
        One object per pulsar, sorted by file path.

    Raises
    ------
    FileNotFoundError
        If no ``*.feather`` files are found in ``data_dir``.
    """
    feather_files = sorted(glob.glob(os.path.join(data_dir, "*.feather")))
    if not feather_files:
        raise FileNotFoundError(f"No *.feather files found in {data_dir}")
    return [LoadWidebandPulsarData.read_feather(f) for f in feather_files]


def compute_joint_grid(pulsars, cadence_days):
    """Compute the common epoch grid and the joint-epoch bins.

    Replicates the grid math of ``check_epoch_alignment.py:195-206`` verbatim so the
    joint-epoch set reproduces the T1.4 verdict: the grid spans the full *union*
    window (``global_min`` -> ``global_max``), TOAs are floor-binned by cadence, and
    a "joint bin" is one occupied by *every* pulsar.

    Parameters
    ----------
    pulsars : list of LoadWidebandPulsarData
    cadence_days : float
        Grid cell width in days.

    Returns
    -------
    tuple
        ``(joint_order, per_pulsar_idx)`` where ``joint_order`` is the sorted list of
        joint bin indices (length = number of joint epochs) and ``per_pulsar_idx`` is
        a list of per-TOA bin-index arrays, one per pulsar (same order as ``pulsars``).
    """
    mjds = [np.asarray(p.toas, dtype=float) / SEC_PER_DAY for p in pulsars]
    grid_start = min(m.min() for m in mjds)
    grid_end = max(m.max() for m in mjds)
    n_bins = int(np.ceil((grid_end - grid_start) / cadence_days)) + 1

    per_pulsar_idx = []
    occupied_sets = []
    for m in mjds:
        idx = np.floor((m - grid_start) / cadence_days).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        per_pulsar_idx.append(idx)
        occupied_sets.append(set(idx.tolist()))

    joint_bins = set.intersection(*occupied_sets) if occupied_sets else set()
    joint_order = sorted(joint_bins)
    return joint_order, per_pulsar_idx


def compute_union_grid(pulsars, cadence_days):
    """Compute the common epoch grid and the *union* of occupied bins.

    The union grid is the full-array analogue of :func:`compute_joint_grid`: a grid
    cell is kept if it is occupied by *any* pulsar (not every pulsar). This preserves
    each pulsar's true baseline instead of truncating the array to the window where
    all pulsars happen to overlap -- essential once the array spans very heterogeneous
    baselines (NG15's short-baseline pulsars would otherwise collapse the intersection
    to ~3 yr). Epochs a given pulsar does not occupy become masked (absent) rows that
    the joint Kalman filter skips in its measurement update.

    Parameters
    ----------
    pulsars : list of LoadWidebandPulsarData
    cadence_days : float

    Returns
    -------
    tuple
        ``(grid_start_mjd, union_order, per_pulsar_idx)``. ``grid_start_mjd`` is the
        grid origin in MJD (so a bin's reference time is
        ``grid_start_mjd + (b + 0.5) * cadence_days``); ``union_order`` is the sorted
        list of occupied bin indices; ``per_pulsar_idx`` is the per-pulsar array of
        per-TOA bin indices (same order as ``pulsars``).
    """
    mjds = [np.asarray(p.toas, dtype=float) / SEC_PER_DAY for p in pulsars]
    grid_start = min(m.min() for m in mjds)
    grid_end = max(m.max() for m in mjds)
    n_bins = int(np.ceil((grid_end - grid_start) / cadence_days)) + 1

    per_pulsar_idx = []
    occupied_sets = []
    for m in mjds:
        idx = np.floor((m - grid_start) / cadence_days).astype(int)
        idx = np.clip(idx, 0, n_bins - 1)
        per_pulsar_idx.append(idx)
        occupied_sets.append(set(idx.tolist()))

    union_bins = set.union(*occupied_sets) if occupied_sets else set()
    union_order = sorted(union_bins)
    return grid_start, union_order, per_pulsar_idx


def drop_dmx_columns(M, fitpars):
    """Drop per-epoch ``DMX_*`` design-matrix columns by fit-parameter name.

    Removes the dispersion-measure-variation nuisance parameters (149-397 of the
    ~165-423 columns per NG15 pulsar). Keeps everything else, including the plain
    ``DM``/``DM1`` astrophysical terms and ``JUMP``/``FD`` instrumental terms.

    Parameters
    ----------
    M : numpy.ndarray
        Design matrix, shape ``(n_rows, n_cols)``.
    fitpars : list of str or None
        Fit-parameter names, one per column.

    Returns
    -------
    tuple
        ``(M_reduced, fitpars_reduced, n_dropped)``.
    """
    if fitpars is None:
        return M, fitpars, 0
    fitpars = list(fitpars)
    keep = [not str(name).upper().startswith("DMX") for name in fitpars]
    keep = np.asarray(keep, dtype=bool)
    if keep.all():
        return M, fitpars, 0
    M_reduced = M[:, keep]
    fitpars_reduced = [fitpars[i] for i in range(len(keep)) if keep[i]]
    return M_reduced, fitpars_reduced, int((~keep).sum())


def drop_zero_columns(M, fitpars):
    """Drop all-zero design-matrix columns (mirrors ``ingest_par_tim.drop_degenerate_columns``).

    After keeping only the joint-epoch rows, a column whose support fell entirely in
    non-joint bins becomes identically zero, which makes ``M^T N^-1 M`` singular and
    ``P_eps`` non-finite. Dropping it is likelihood-preserving (a zero column adds
    nothing to ``M beta``).

    Parameters
    ----------
    M : numpy.ndarray
        Design matrix, shape ``(n_rows, n_cols)``.
    fitpars : list of str or None

    Returns
    -------
    tuple
        ``(M_reduced, fitpars_reduced, n_dropped)``.
    """
    keep = np.sqrt(np.sum(M**2, axis=0)) > 0
    if keep.all():
        return M, fitpars, 0
    M_reduced = M[:, keep]
    if fitpars is not None:
        fitpars_reduced = [fitpars[i] for i in range(len(keep)) if keep[i]]
    else:
        fitpars_reduced = None
    return M_reduced, fitpars_reduced, int((~keep).sum())


def bin_pulsar(pulsar, bin_idx, joint_order):
    """Inverse-variance epoch-average one pulsar onto the joint grid.

    Parameters
    ----------
    pulsar : LoadWidebandPulsarData
    bin_idx : numpy.ndarray
        Per-TOA bin indices for this pulsar (from :func:`compute_joint_grid`).
    joint_order : list of int
        Sorted joint bin indices; one output row is produced per entry, in order.

    Returns
    -------
    tuple
        ``(toas_b, res_b, err_b, M_b)`` with the first three of shape ``(nepoch,)``
        (seconds) and ``M_b`` of shape ``(nepoch, n_cols)``. ``nepoch = len(joint_order)``.
    """
    toas = np.asarray(pulsar.toas, dtype=float)
    res = np.asarray(pulsar.residuals, dtype=float)
    err = np.asarray(pulsar.toaerrs, dtype=float)
    M = np.asarray(pulsar.M_matrix, dtype=float)

    nepoch = len(joint_order)
    toas_b = np.empty(nepoch)
    res_b = np.empty(nepoch)
    err_b = np.empty(nepoch)
    M_b = np.empty((nepoch, M.shape[1]))

    for k, b in enumerate(joint_order):
        sel = np.flatnonzero(bin_idx == b)
        # A joint bin is occupied by every pulsar, so `sel` is never empty.
        w_raw = 1.0 / err[sel] ** 2
        w = w_raw / w_raw.sum()
        toas_b[k] = np.dot(w, toas[sel])
        res_b[k] = np.dot(w, res[sel])
        err_b[k] = 1.0 / np.sqrt(w_raw.sum())
        M_b[k, :] = w @ M[sel, :]

    return toas_b, res_b, err_b, M_b


def bin_pulsar_union(pulsar, bin_idx, grid_start, grid_order, cadence_days):
    """Inverse-variance epoch-average one pulsar onto the union grid, with a mask.

    Unlike :func:`bin_pulsar`, the grid contains cells this pulsar does not occupy.
    Occupied cells are inverse-variance averaged exactly as in the intersection case;
    unoccupied cells become masked (absent) rows with a zero residual, a zero design
    row (so ``P_eps`` is computed from the present rows only) and a placeholder error.
    Every row's reference TOA is the **grid-cell centre** (not the mean of the TOAs in
    it), which is well defined for absent cells and guarantees a strictly increasing,
    correctly spaced time axis for the filter's ``dt`` sequence.

    Returns
    -------
    tuple
        ``(toas_b, res_b, err_b, M_b, mask_b)`` -- the first three shape ``(nepoch,)``
        (seconds), ``M_b`` shape ``(nepoch, n_cols)``, ``mask_b`` shape ``(nepoch,)``
        with 1.0 for occupied cells and 0.0 for absent ones.
    """
    toas = np.asarray(pulsar.toas, dtype=float)
    res = np.asarray(pulsar.residuals, dtype=float)
    err = np.asarray(pulsar.toaerrs, dtype=float)
    M = np.asarray(pulsar.M_matrix, dtype=float)

    nepoch = len(grid_order)
    toas_b = np.empty(nepoch)
    res_b = np.zeros(nepoch)
    err_b = np.empty(nepoch)
    M_b = np.zeros((nepoch, M.shape[1]))
    mask_b = np.zeros(nepoch)

    # Placeholder error for absent rows (masked out; kept at the present-error scale so
    # the R matrix stays well conditioned even though those rows never enter the fit).
    placeholder_err = float(np.median(err))

    for k, b in enumerate(grid_order):
        toas_b[k] = (grid_start + (b + 0.5) * cadence_days) * SEC_PER_DAY
        sel = np.flatnonzero(bin_idx == b)
        if sel.size:
            w_raw = 1.0 / err[sel] ** 2
            w = w_raw / w_raw.sum()
            res_b[k] = np.dot(w, res[sel])
            err_b[k] = 1.0 / np.sqrt(w_raw.sum())
            M_b[k, :] = w @ M[sel, :]
            mask_b[k] = 1.0
        else:
            err_b[k] = placeholder_err

    return toas_b, res_b, err_b, M_b, mask_b


def build_aligned_object(pulsar, toas_b, res_b, err_b, M_b, fitpars_b):
    """Wrap binned arrays in a namespace and build a LoadWidebandPulsarData.

    Mirrors ``read_feather``'s SimpleNamespace -> ``__init__`` path (data_loader.py
    :613), so ``M_scaled`` and ``P_eps`` are recomputed on the reduced binned matrix.

    Returns
    -------
    LoadWidebandPulsarData
    """
    ns = types.SimpleNamespace(
        toas=toas_b,
        toaerrs=err_b,
        residuals=res_b,
        fitpars=fitpars_b,
        Mmat=M_b,
        name=pulsar.name,
        _raj=pulsar.RA,
        _decj=pulsar.DEC,
        _pdist=(pulsar.distance_kpc, pulsar.distance_err_kpc),
    )
    obj = LoadWidebandPulsarData(ns)
    obj.F0 = getattr(pulsar, "F0", None)
    return obj


def condition_number(obj):
    """Return cond(M_scaled^T N^-1 M_scaled) for the timing marginalization."""
    err = np.asarray(obj.toaerrs, dtype=float)
    Ninv = np.diag(1.0 / err**2)
    MtNinvM = obj.M_scaled.T @ Ninv @ obj.M_scaled
    return float(np.linalg.cond(MtNinvM))


def process_all(pulsars, cadence_days, grid="intersection"):
    """Bin, reduce and build an aligned object for every pulsar.

    Parameters
    ----------
    pulsars : list of LoadWidebandPulsarData
    cadence_days : float
    grid : {"intersection", "union"}
        ``intersection`` keeps only epochs occupied by every pulsar (the original
        subset behaviour, no mask). ``union`` keeps every occupied epoch and emits a
        per-pulsar mask so the joint filter skips absent epochs -- the full-array path.

    Returns
    -------
    tuple
        ``(built, rows)`` where ``built`` is a list of ``(aligned_object, F0)`` and
        ``rows`` is a list of per-pulsar summary dicts. In ``union`` mode each object
        carries a ``mask`` attribute consumed by :meth:`save_feather`.
    """
    if grid == "union":
        grid_start, order, per_pulsar_idx = compute_union_grid(pulsars, cadence_days)
    else:
        order, per_pulsar_idx = compute_joint_grid(pulsars, cadence_days)
        grid_start = None
    nepoch = len(order)

    label = "union" if grid == "union" else "joint"
    print(f"Common {cadence_days:g}-day grid ({grid}): {label} epochs = {nepoch}")
    if grid == "intersection":
        if nepoch < MIN_VIABLE_JOINT_EPOCHS:
            raise SystemExit(
                f"*** STOP -- FLAG FOR REVIEW (RISK A): only {nepoch} joint epochs "
                f"(< {MIN_VIABLE_JOINT_EPOCHS} floor). Joint alignment is not viable "
                "for this subset/cadence. Consider --grid union. ***"
            )
        if nepoch != EXPECTED_JOINT_EPOCHS:
            print(
                f"  WARNING: expected {EXPECTED_JOINT_EPOCHS} joint epochs (T1.4), "
                f"got {nepoch} -- grid inputs may have changed."
            )
    print()

    built = []
    rows = []
    for pulsar, bin_idx in zip(pulsars, per_pulsar_idx):
        n_toas_orig = int(np.asarray(pulsar.toas).size)
        n_cols_orig = int(np.asarray(pulsar.M_matrix).shape[1])

        # Drop DMX nuisance columns (by name) before binning -- cheaper, and the
        # choice is independent of the row grid.
        M0, fitpars0, n_dmx = drop_dmx_columns(
            np.asarray(pulsar.M_matrix, dtype=float), pulsar.fitpars
        )
        reduced = types.SimpleNamespace(
            toas=pulsar.toas,
            residuals=pulsar.residuals,
            toaerrs=pulsar.toaerrs,
            M_matrix=M0,
            name=pulsar.name,
        )
        if grid == "union":
            toas_b, res_b, err_b, M_b, mask_b = bin_pulsar_union(
                reduced, bin_idx, grid_start, order, cadence_days
            )
        else:
            toas_b, res_b, err_b, M_b = bin_pulsar(reduced, bin_idx, order)
            mask_b = None

        # Now drop any column that lost all support in the kept bins.
        M_b, fitpars_b, n_zero = drop_zero_columns(M_b, fitpars0)
        n_cols_final = M_b.shape[1]

        n_present = int(mask_b.sum()) if mask_b is not None else nepoch
        n_dropped_toas = n_toas_orig - int(np.isin(bin_idx, order).sum())

        obj = build_aligned_object(pulsar, toas_b, res_b, err_b, M_b, fitpars_b)
        if mask_b is not None:
            obj.mask = mask_b

        # Correctness guards.
        if not np.all(np.isfinite(obj.P_eps)):
            raise SystemExit(
                f"*** {pulsar.name}: P_eps is non-finite after reduction "
                f"({n_cols_final} cols on {n_present} present epochs). Enable the SVD "
                "conditioning fallback (see module docstring). ***"
            )
        if not np.all(np.diff(toas_b) > 0):
            raise SystemExit(
                f"*** {pulsar.name}: binned TOAs are not strictly increasing; "
                "row ordering is broken. ***"
            )
        cond = condition_number(obj)
        cond_flag = "  (STIFF)" if cond > COND_WARN_THRESHOLD else ""

        built.append((obj, getattr(pulsar, "F0", None)))
        rows.append(
            {
                "name": pulsar.name,
                "n_toas_orig": n_toas_orig,
                "nepoch": nepoch,
                "n_present": n_present,
                "n_toas_dropped": n_dropped_toas,
                "n_cols_orig": n_cols_orig,
                "n_dmx": n_dmx,
                "n_zero": n_zero,
                "n_cols_final": n_cols_final,
                "peps_finite": True,
                "cond": cond,
                "cond_flag": cond_flag,
                "err_min": float(err_b.min()),
                "err_med": float(np.median(err_b)),
            }
        )
    return built, rows


def write_aligned(built, out_dir, overwrite):
    """Write each aligned object to ``out_dir`` via the stock ``save_feather`` schema."""
    os.makedirs(out_dir, exist_ok=True)
    for obj, f0 in built:
        path = os.path.join(out_dir, f"{obj.name}.feather")
        if os.path.exists(path) and not overwrite:
            raise SystemExit(f"*** {path} exists; pass --overwrite to replace it. ***")
        obj.save_feather(path, F0=f0)
        print(f"  wrote {path}")


def verify(out_dir, n_pulsars):
    """Assert the stock GWB path consumes the aligned feathers cleanly (T1.5 gate)."""
    print("\nVerifying get_processed_residuals(mode='gwb') on the aligned dir ...")
    data = LoadWidebandPulsarData.get_processed_residuals(out_dir, mode="gwb")
    pr = data["processed_residuals"]
    res = np.asarray(pr["residuals"])
    errs = np.asarray(pr["errors"])
    toas = np.asarray(pr["toas"])
    hd = np.asarray(data["hd_correlation"])

    nepoch = res.shape[0]
    assert res.shape == (nepoch, n_pulsars), f"residuals shape {res.shape}"
    assert errs.shape == (nepoch, n_pulsars), f"errors shape {errs.shape}"
    assert toas.shape == (nepoch,), f"toas shape {toas.shape}"
    assert hd.shape == (n_pulsars, n_pulsars), f"hd shape {hd.shape}"
    assert np.allclose(np.diag(hd), 1.0), "HD diagonal is not unit"
    assert np.all(np.isfinite(res)) and np.all(
        np.isfinite(errs)
    ), "non-finite residuals/errors"

    mask_msg = ""
    if "mask" in pr:
        mask = np.asarray(pr["mask"])
        assert mask.shape == (nepoch, n_pulsars), f"mask shape {mask.shape}"
        assert set(np.unique(mask)).issubset({0.0, 1.0}), "mask is not binary"
        frac = float(mask.mean())
        mask_msg = f", mask {mask.shape} ({frac*100:.0f}% cells present)"

    print(
        f"  OK: residuals {res.shape}, errors {errs.shape}, toas {toas.shape}, "
        f"HD {hd.shape} (unit diagonal){mask_msg}. "
        "process_pulsar_residuals_by_epoch no longer raises."
    )


def print_summary(rows):
    """Print the per-pulsar attrition/conditioning table."""
    print("\n" + "=" * 100)
    print("PER-PULSAR SUMMARY (attrition is explicit, not silent)")
    print("=" * 100)
    header = (
        f"{'pulsar':<12} {'TOAs':>6} {'nepoch':>6} {'present':>7} {'TOAdrop':>8} "
        f"{'cols':>5} {'DMX':>5} {'zero':>5} {'->cols':>6} {'Peps':>5} "
        f"{'cond':>9} {'err_med(us)':>11}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{r['name']:<12} {r['n_toas_orig']:>6d} {r['nepoch']:>6d} "
            f"{r['n_present']:>7d} "
            f"{r['n_toas_dropped']:>8d} {r['n_cols_orig']:>5d} {r['n_dmx']:>5d} "
            f"{r['n_zero']:>5d} {r['n_cols_final']:>6d} "
            f"{'fin' if r['peps_finite'] else 'BAD':>5} "
            f"{r['cond']:>9.1e}{r['cond_flag']} {r['err_med'] * 1e6:>11.3f}"
        )
    print(
        "\nColumns: TOAs->rows = original TOA count binned to joint epochs; "
        "TOAdrop = TOAs falling in non-joint bins;\n"
        "  cols->cols = design-matrix columns before/after DMX + all-zero drop; "
        "cond = cond(M^T N^-1 M).\n"
    )


def run(data_dir, out_dir, cadence_days, overwrite, grid="intersection"):
    """Load, bin+reduce, write, and verify the aligned feathers."""
    print(f"Reading ragged feathers from: {data_dir}\n")
    pulsars = load_pulsars(data_dir)
    print(f"Loaded {len(pulsars)} pulsars: {', '.join(p.name for p in pulsars)}\n")

    built, rows = process_all(pulsars, cadence_days, grid=grid)
    print(f"Writing aligned feathers to: {out_dir}")
    write_aligned(built, out_dir, overwrite)
    print_summary(rows)
    verify(out_dir, len(pulsars))
    print("\nAligned feathers built and consumed by the GWB path.")


def main():
    """Parse command-line arguments and build the aligned feathers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory of ingested ragged *.feather files (default: the workflow data/ dir)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for aligned feathers (default: <data-dir>/aligned)",
    )
    parser.add_argument(
        "--cadence",
        type=float,
        default=DEFAULT_CADENCE_DAYS,
        help="Common-grid cadence in days (default: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing aligned feathers",
    )
    parser.add_argument(
        "--grid",
        choices=["intersection", "union"],
        default="intersection",
        help=(
            "Epoch-grid strategy. 'intersection' (default) keeps only epochs occupied "
            "by every pulsar (the 6-pulsar subset behaviour). 'union' keeps every "
            "occupied epoch and emits a per-pulsar observation mask so the joint filter "
            "skips absent epochs -- required to scale to the full heterogeneous array."
        ),
    )
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.data_dir, "aligned")
    run(args.data_dir, out_dir, args.cadence, args.overwrite, grid=args.grid)


if __name__ == "__main__":
    main()
