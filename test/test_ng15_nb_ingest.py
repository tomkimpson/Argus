"""NG15 narrowband ingest round-trips the Discovery feathers (spec task 2.1).

Skipped when the feathers are absent (CI has no NG15 data). Point NG15_DISCOVERY_DIR at
a Discovery checkout to run them.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflows" / "ng15_nb" / "compression"))
import ingest  # noqa: E402

DISCOVERY_DIR = Path(
    os.environ.get("NG15_DISCOVERY_DIR", "/fred/oz022/tkimpson/external/discovery")
)
MANIFEST = json.loads((REPO / "data/ng15_nb/MANIFEST.json").read_text())
F0 = json.loads((REPO / "data/ng15_nb/f0_catalog.json").read_text())["f0_hz"]
PULSARS = ["B1855+09", "J1909-3744"]


def _feather(psr):
    entry = next(e for e in MANIFEST["feathers"] if e["pulsar"] == psr)
    path = DISCOVERY_DIR / "data" / entry["file"]
    if not path.exists():
        pytest.skip(f"NG15 feathers not found under {DISCOVERY_DIR}")
    return path


def _discovery_ecorr_umat(toas, backend):
    """Discovery's makegp_ecorr (enterprise=False) U-matrix, transcribed verbatim."""
    backends = [b for b in sorted(set(backend)) if b != ""]
    masks = [np.array(backend == b) for b in backends]
    umats = []
    for mask in masks:
        first_valid_bin = 0 if len(np.unique(masks)) == 1 else 1
        bins = ingest._quantize(toas * mask)
        umats.append(
            np.vstack([bins == i for i in range(first_valid_bin, bins.max() + 1)]).T
        )
    return np.hstack(umats)


@pytest.mark.parametrize("psr", PULSARS)
def test_round_trip(psr, tmp_path):
    path = _feather(psr)
    cache = ingest.ingest(path, F0[psr])
    back = ingest.load(ingest.save(cache, tmp_path))

    table = feather.read_table(str(path))
    mcols = [c for c in table.column_names if c.startswith("Mmat_")]
    assert back["toas"].size == table.num_rows
    assert sorted(back["M_names"].tolist()) == sorted(mcols)
    assert back["M"].shape == (table.num_rows, len(mcols))
    for j, name in enumerate(back["M_names"]):
        np.testing.assert_array_equal(back["M"][:, j], table.column(name).to_numpy())
    for key, col in [("toas", "toas"), ("residuals", "residuals"), ("freqs", "freqs")]:
        np.testing.assert_array_equal(back[key], table.column(col).to_numpy())
    assert back["name"] == psr

    # Every DMX table entry is one DMX column, and every TOA lies in exactly one window.
    meta = json.loads(table.schema.metadata[b"json"])
    assert back["is_dmx"].sum() == len(meta["dmx"])
    assert (back["dmx_window"] >= 0).all()
    assert sorted(back["dmx_window_label"].tolist()) == sorted(meta["dmx"])


@pytest.mark.parametrize("psr", PULSARS)
def test_ecorr_epochs_match_discovery(psr):
    cache = ingest.ingest(_feather(psr), F0[psr])
    U = _discovery_ecorr_umat(cache["toas"], cache["backend"])
    ours = np.zeros_like(U)
    ours[np.arange(cache["toas"].size), cache["ecorr_epoch"]] = True
    np.testing.assert_array_equal(ours, U)
    assert cache["ecorr_var"].size == U.shape[1]


def test_white_noise_t2equad():
    """EFAC²(σ² + EQUAD²), checked for one backend by hand."""
    psr = PULSARS[1]
    cache = ingest.ingest(_feather(psr), F0[psr])
    nd = ingest.read_feather(_feather(psr))["noisedict"]
    b = "Rcvr1_2_GUPPI"
    sel = cache["backend"] == b
    efac, eq = nd[f"{psr}_{b}_efac"], 10 ** nd[f"{psr}_{b}_log10_t2equad"]
    np.testing.assert_allclose(
        cache["white_var"][sel], efac**2 * (cache["toaerrs"][sel] ** 2 + eq**2)
    )
