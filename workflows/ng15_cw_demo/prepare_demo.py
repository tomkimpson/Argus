"""Prepare the NG15 CW injection-recovery demo.

Two steps:

1. Ingest a chosen subset of NanoGrav *Discovery* NG15 feathers into Argus-native
   feathers + fixed-noise files, via ``scripts/ingest_discovery_feather.py``.
2. Inject a known Earth-term continuous-wave (CW) signal (computed with Argus's own
   waveform) into each pulsar's residuals and re-save, recording the true injected
   parameters to ``injection.json``.

The result is a self-contained data directory the standard Argus CW workflow
(``run_analysis.py`` + ``configs/ng15_cw_injection.ini``) can recover from.

Usage
-----
    python prepare_demo.py \
        --discovery-dir /path/to/discovery/data \
        --out-dir ./demo_data \
        [--pulsars J1909-3744 J1744-1134 ...]

See ../../docs/discovery_review.md for the data/noise/convention caveats.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

from argus.data_loader import LoadWidebandPulsarData
from argus.gravitational_waves import (
    antenna_pattern_single,
    compute_cw_signal_single_pulsar,
)

# Default injection: a single loud-ish CW (Earth term only) well inside the band.
DEFAULT_INJECTION = {
    "alpha_gw": 2.0,      # source RA (rad)
    "delta_gw": 0.3,      # source DEC (rad)
    "f_gw": 2.0e-8,       # GW frequency (Hz) ~ 20 nHz
    "h0": 5.0e-14,        # strain amplitude
    "cos_iota": 0.3,
    "psi": 0.7,           # NOTE Argus psi convention (= -psi_discovery)
    "Phi0": 1.0,
}

DEFAULT_PULSARS = [
    "J1909-3744", "J1744-1134", "J2317+1439", "J0030+0451",
    "J0613-0200", "J1600-3053",
]


def inject_cw(out_dir: str, injection: dict) -> None:
    """Add an Earth-term CW to each Argus feather's residuals, in place."""
    feathers = sorted(
        f for f in os.listdir(out_dir) if f.endswith(".feather")
    )
    for fname in feathers:
        path = os.path.join(out_dir, fname)
        psr = LoadWidebandPulsarData.read_feather(path)
        Fp, Fc = antenna_pattern_single(
            psr.RA, psr.DEC, injection["alpha_gw"], injection["delta_gw"], injection["psi"]
        )
        cw = compute_cw_signal_single_pulsar(
            jnp.asarray(psr.toas), injection["f_gw"], injection["h0"],
            injection["cos_iota"], injection["Phi0"], Fp, Fc,
            pulsar_distance=0.0, geometric_factor=0.0,
        )
        psr.residuals = np.asarray(psr.residuals) + np.asarray(cw)
        psr.save_feather(path, F0=getattr(psr, "F0", None))
        print(f"  injected CW into {psr.name} (rms add = {np.std(np.asarray(cw)):.2e} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discovery-dir", required=True, help="Discovery data/ dir (v1p1_*.feather)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "demo_data"))
    ap.add_argument("--pulsars", nargs="*", default=DEFAULT_PULSARS)
    ap.add_argument("--no-inject", action="store_true", help="Ingest only; skip CW injection (real-data mode).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: ingest via the adapter.
    adapter = os.path.join(REPO, "scripts", "ingest_discovery_feather.py")
    cmd = [sys.executable, adapter, args.discovery_dir, args.out_dir, "--pulsars", *args.pulsars]
    print("Ingesting Discovery feathers ->", args.out_dir)
    subprocess.run(cmd, check=True)

    # Step 2: inject CW (unless real-data mode).
    if args.no_inject:
        print("Real-data mode: no CW injected.")
        return
    print("\nInjecting Earth-term CW:")
    inject_cw(args.out_dir, DEFAULT_INJECTION)
    with open(os.path.join(args.out_dir, "injection.json"), "w") as f:
        json.dump(DEFAULT_INJECTION, f, indent=2)
    print(f"\nDemo data ready in {args.out_dir} (true params in injection.json)")


if __name__ == "__main__":
    main()
