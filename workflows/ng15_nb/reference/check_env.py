"""Smoke-test the `discovery` env: evaluate J1909-3744's published single-pulsar likelihood.

Spec task 1.2. Builds NANOGrav's single-pulsar model (fixed per-backend EFAC/EQUAD and
ECORR from the feather's noise dictionary, the timing model marginalised under the 1e40
improper prior, 30-mode power-law red noise) and evaluates it at the published red-noise
values. Checks the value is finite, that jit and eager agree, and that the gradient is
finite. Also re-verifies the Discovery checkout against data/ng15_nb/MANIFEST.json.

The feathers store ``<psr>_red_noise_gamma`` with the sign flipped (all 25 pulsars that
carry one have γ < 0, outside NANOGrav's U[0, 7] prior; the magnitudes are the
published spectral indices, e.g. 0.91 for the shallow J1012+5307). The GWB models sample
red noise and take only the white noise from the dictionary, so this affects nothing
downstream; here the published point is evaluated at γ = −stored. (At J1909 that lifts
lnL by ~700 nats and leaves a near-zero gradient, i.e. it lands at the published MAP.)

    conda activate discovery
    python workflows/ng15_nb/reference/check_env.py <DISCOVERY_DIR>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import discovery as ds  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
PSR = "J1909-3744"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("discovery_dir")
    args = ap.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "workflows/ng15_nb/scripts/build_manifest.py"),
            args.discovery_dir,
            "--check",
        ],
        check=True,
    )
    manifest = json.loads((REPO_ROOT / "data/ng15_nb/MANIFEST.json").read_text())
    (entry,) = [e for e in manifest["feathers"] if e["pulsar"] == PSR]

    psr = ds.Pulsar.read_feather(str(Path(args.discovery_dir) / "data" / entry["file"]))
    model = ds.PulsarLikelihood(
        [
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_ecorr(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True),
            ds.makegp_fourier(
                psr, ds.powerlaw, 30, T=ds.getspan(psr), name="red_noise"
            ),
        ]
    )
    logl = model.logL
    params = {p: psr.noisedict[p] for p in logl.params}
    stored_gamma = params[f"{PSR}_red_noise_gamma"]
    assert (
        stored_gamma < 0
    ), "feather red_noise_gamma is no longer sign-flipped; drop the correction"
    params[f"{PSR}_red_noise_gamma"] = -stored_gamma

    eager = float(logl(params))
    jitted = float(jax.jit(logl)(params))
    grad = jax.grad(logl)(params)

    print(
        f"discovery {ds.__file__}  jax {jax.__version__}  backend {jax.default_backend()}"
    )
    print(
        f"{PSR}: {len(psr.toas)} TOAs, M {psr.Mmat.shape}, backends {sorted(set(psr.backend_flags))}"
    )
    print(f"params at published values: {params}")
    print(
        f"lnL eager = {eager:.6f}   jit = {jitted:.6f}   |diff| = {abs(eager - jitted):.2e}"
    )
    print(f"grad: {{{', '.join(f'{k}: {float(v):.4e}' for k, v in grad.items())}}}")

    assert np.isfinite(eager), "lnL not finite"
    assert abs(eager - jitted) <= 1e-8 * abs(eager), "jit and eager disagree"
    assert all(np.isfinite(float(v)) for v in grad.values()), "gradient not finite"
    print("OK")


if __name__ == "__main__":
    main()
