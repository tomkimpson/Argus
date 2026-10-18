"""Narrowband NG15 ingest: Discovery feathers -> per-TOA arrays for epoch compression.

Spec task 2.1 (spec/spec.md, D1/D2). The output keeps everything compression needs at TOA
level, with no folding:

- TOAs (s), residuals (s), raw TOA errors (s) and radio frequencies (MHz);
- the backend flag of each TOA, and its fixed white-noise variance under the published
  noise dictionary in Discovery's t2equad convention, EFAC_b² (σ² + EQUAD_b²);
- the ECORR epoch of each TOA and each epoch's fixed ECORR variance, built exactly as
  ``discovery.makegp_ecorr`` (enterprise=False) builds them: per backend, TOAs within
  dt = 1 s of an epoch's first TOA share the epoch, and singletons are epochs too;
- the full design matrix with the feather's own column names (``Mmat_j``), and the DMX
  window of each TOA.

DMX windows are read off the design matrix, not off the DMXR1/DMXR2 ranges in the feather
metadata. The ranges are topocentric MJDs while the TOAs are barycentric, so a TOA near a
window edge can fall outside its own range (J1713+0747 has 91 such TOAs, and one window of
zero width). The design matrix is what the likelihood uses, so it is the authority: a DMX
column is a local column (nonzero on a small subset of TOAs) whose values are ∝ 1/f²
there. Each DMX column is then labelled with its DMX table entry (matched in time order); the label is
informational.

pyarrow only; no discovery/enterprise import.
"""

import json
import re
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ECORR_DT = 1.0  # s; discovery.signals.quantize default
DMX_RTOL = 1e-6  # tolerance on M·f² being constant over a DMX column's support
LOCAL_FRAC = 0.5  # a column nonzero on fewer than this fraction of TOAs is "local"


def read_feather(path):
    """Read one Discovery feather into a dict of numpy arrays (pyarrow only)."""
    table = feather.read_table(str(path))
    names = table.column_names
    meta = json.loads(table.schema.metadata[b"json"])
    mcols = sorted(
        (c for c in names if re.fullmatch(r"Mmat_\d+", c)),
        key=lambda c: int(c.split("_")[1]),
    )
    return {
        "name": meta["name"],
        "toas": table.column("toas").to_numpy().astype(np.float64),
        "toaerrs": table.column("toaerrs").to_numpy().astype(np.float64),
        "residuals": table.column("residuals").to_numpy().astype(np.float64),
        "freqs": table.column("freqs").to_numpy().astype(np.float64),
        "backend": table.column("backend_flags").to_numpy().astype("U"),
        "M": np.column_stack([table.column(c).to_numpy() for c in mcols]).astype(
            np.float64
        ),
        "M_names": np.array(mcols),
        "phi": float(meta["phi"]),
        "theta": float(meta["theta"]),
        "noisedict": meta["noisedict"],
        "dmx": meta["dmx"],
    }


def white_noise_variance(raw):
    """Per-TOA fixed white-noise variance, EFAC_b² (σ² + EQUAD_b²) (t2equad)."""
    nd, name = raw["noisedict"], raw["name"]
    var = np.full(raw["toas"].size, np.nan)
    for b in np.unique(raw["backend"]):
        sel = raw["backend"] == b
        efac = nd[f"{name}_{b}_efac"]
        equad2 = 10.0 ** (2.0 * nd[f"{name}_{b}_log10_t2equad"])
        var[sel] = efac**2 * (raw["toaerrs"][sel] ** 2 + equad2)
    return var


def _quantize(toas, dt=ECORR_DT):
    """Verbatim port of ``discovery.signals.quantize``."""
    isort = np.argsort(toas)
    bins = np.zeros_like(toas, np.int64)
    b, v = 0, toas.min()
    for j in isort:
        if toas[j] - v > dt:
            v = toas[j]
            b = b + 1
        bins[j] = b
    return bins


def ecorr_epochs(raw):
    """ECORR epoch id per TOA and fixed ECORR variance per epoch.

    Equivalent to ``makegp_ecorr(psr, noisedict)`` with enterprise=False: Discovery
    quantizes ``toas * mask`` per backend and skips the bin of masked-out zeros, which is
    the same as quantizing each backend's own TOAs. Epoch ids are contiguous, ordered by
    backend (sorted) then time, as Discovery orders its U-matrix columns.
    """
    nd, name = raw["noisedict"], raw["name"]
    epoch = np.full(raw["toas"].size, -1, dtype=np.int64)
    variances = []
    for b in sorted(set(raw["backend"]) - {""}):
        sel = np.nonzero(raw["backend"] == b)[0]
        bins = _quantize(raw["toas"][sel])
        _, inv = np.unique(bins, return_inverse=True)
        epoch[sel] = len(variances) + inv
        var = 10.0 ** (2.0 * nd[f"{name}_{b}_log10_ecorr"])
        variances.extend([var] * (inv.max() + 1))
    if (epoch < 0).any():
        raise ValueError(f"{name}: TOAs with an empty backend flag")
    return epoch, np.array(variances)


def classify_columns(raw):
    """Split M into DMX columns and the rest; return the DMX window id of each TOA.

    Returns (is_dmx bool per column, window id per TOA or -1, window->column index).
    Raises if two DMX columns overlap, which would make windows ill-defined.
    """
    M, f = raw["M"], raw["freqs"]
    nz = M != 0.0
    frac = nz.mean(axis=0)
    is_dmx = np.zeros(M.shape[1], dtype=bool)
    for j in np.nonzero(frac < LOCAL_FRAC)[0]:
        s = nz[:, j]
        r = M[s, j] * f[s] ** 2
        if np.ptp(r) <= DMX_RTOL * np.abs(r).mean():
            is_dmx[j] = True
    window = np.full(M.shape[0], -1, dtype=np.int64)
    win_col = np.nonzero(is_dmx)[0]
    for w, j in enumerate(win_col):
        s = nz[:, j]
        if (window[s] >= 0).any():
            raise ValueError(f"{raw['name']}: DMX column {j} overlaps another window")
        window[s] = w
    return is_dmx, window, win_col


def label_dmx_columns(raw, win_col, window):
    """Name each DMX column after its DMX table entry, matched in time order.

    Windows are disjoint and there is one table entry per column, so sorting both by time
    pairs them one-to-one. Each pair is checked: the column's TOAs must lie within a day
    of the entry's DMXR range (topocentric vs barycentric time differ by < 0.01 d).
    """
    entries = sorted(raw["dmx"].items(), key=lambda kv: kv[1]["DMXR1"])
    if len(entries) != len(win_col):
        raise ValueError(
            f"{raw['name']}: {len(win_col)} DMX columns vs {len(entries)} table entries"
        )
    t_mid = np.array(
        [np.mean(raw["toas"][window == w]) / 86400.0 for w in range(len(win_col))]
    )
    labels = np.empty(len(win_col), dtype=object)
    for w, (key, v) in zip(np.argsort(t_mid), entries):
        t = raw["toas"][window == w] / 86400.0
        if t.min() < v["DMXR1"] - 1.0 or t.max() > v["DMXR2"] + 1.0:
            raise ValueError(f"{raw['name']}: DMX column {w} does not match {key}")
        labels[w] = key
    return labels.astype("U")


def ingest(path, f0):
    """Ingest one feather. Returns the per-TOA cache dict (all numpy)."""
    raw = read_feather(path)
    epoch, ecorr_var = ecorr_epochs(raw)
    is_dmx, window, win_col = classify_columns(raw)
    return {
        "name": raw["name"],
        "f0": float(f0),
        "phi": raw["phi"],
        "theta": raw["theta"],
        "toas": raw["toas"],
        "residuals": raw["residuals"],
        "toaerrs": raw["toaerrs"],
        "freqs": raw["freqs"],
        "backend": raw["backend"],
        "white_var": white_noise_variance(raw),
        "ecorr_epoch": epoch,
        "ecorr_var": ecorr_var,
        "dmx_window": window,
        "M": raw["M"],
        "M_names": raw["M_names"],
        "is_dmx": is_dmx,
        "dmx_window_col": win_col,
        "dmx_window_label": label_dmx_columns(raw, win_col, window),
    }


def save(cache, out_dir):
    out = Path(out_dir) / f"{cache['name']}.npz"
    np.savez(out, **cache)
    return out


def load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in ("name",):
        d[k] = str(d[k])
    for k in ("f0", "phi", "theta"):
        d[k] = float(d[k])
    return d


def summarize(cache):
    """The per-pulsar facts task 2.1 reports (before any compression rule is applied)."""
    toas, win, ep, f = (
        cache["toas"],
        cache["dmx_window"],
        cache["ecorr_epoch"],
        cache["freqs"],
    )
    nwin = len(cache["dmx_window_col"])
    lengths = np.array(
        [np.ptp(toas[win == w]) / 86400.0 for w in range(nwin)], dtype=float
    )
    single_freq = sum(np.unique(f[win == w]).size == 1 for w in range(nwin))
    multi_ecorr = sum(np.unique(ep[win == w]).size > 1 for w in range(nwin))
    straddle = sum(np.unique(win[ep == e]).size > 1 for e in range(ep.max() + 1))
    return {
        "pulsar": cache["name"],
        "n_toa": int(toas.size),
        "n_cols": int(cache["M"].shape[1]),
        "n_dmx_cols": nwin,
        "n_other_cols": int((~cache["is_dmx"]).sum()),
        "n_toa_without_dmx": int((win < 0).sum()),
        "dmx_window_days": {
            "min": float(lengths.min()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": float(lengths.max()),
        },
        "n_single_freq_windows": int(single_freq),
        "n_windows_multi_ecorr": int(multi_ecorr),
        "n_ecorr_epochs": int(ep.max() + 1),
        "n_ecorr_straddling": int(straddle),
        "tspan_yr": float(np.ptp(toas) / (365.25 * 86400.0)),
    }
