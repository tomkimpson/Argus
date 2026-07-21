"""T3.4 CURN driver — run the production NG15 pipeline with the Hellings-Downs
inter-pulsar correlation replaced by the IDENTITY matrix (common uncorrelated red
noise). This is the null against which the HD run's Bayes factor is measured.

The ONLY difference from run_analysis.py is a runtime override, applied with NO
library edit (per repo convention): the data loader's ``hd_correlation`` matrix
(an Npsr x Npsr matrix built by ``gravitational_waves.hellings_downs``; verified
diagonal = 1) is replaced by ``np.eye(Npsr)`` before the Kalman filter consumes it
(jax_kalman_filter.py:380). Identity keeps each pulsar's auto-power (the common
amplitude) and zeros only the cross-correlations => exactly CURN. Everything else
— priors, NUTS settings, noise handling, saving/plots — is the untouched production
path, so logZ_HD and logZ_CURN are comparable term-for-term.

Point it at a config whose output_id is distinct from the HD run (ng15_curn):
    python run_curn.py configs/ng15_curn_config.ini
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import jax

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_root, "python"))

from argus import utils, workflow, data_loader  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _install_curn_override():
    """Monkeypatch the data loader so hd_correlation becomes the identity (CURN)."""
    _orig = data_loader.LoadWidebandPulsarData.get_processed_residuals

    def _patched(directory, excluded_psrs=[], mode="gwb"):
        data = _orig(directory, excluded_psrs=excluded_psrs, mode=mode)
        hd = data.get("hd_correlation")
        if hd is not None:
            n = np.asarray(hd).shape[0]
            data["hd_correlation"] = np.eye(n)
            print(
                f"[CURN] Overrode hd_correlation with identity ({n}x{n}) — "
                f"common uncorrelated red noise (cross-correlations zeroed)."
            )
        else:
            raise RuntimeError("[CURN] hd_correlation is None; expected gwb mode.")
        return data

    data_loader.LoadWidebandPulsarData.get_processed_residuals = staticmethod(_patched)


def main():
    parser = argparse.ArgumentParser(description="Run CURN (identity-ORF) inference.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    print("=== NG15 SGWB WORKFLOW (CURN — identity ORF) ===")
    print(f"JAX version: {jax.__version__}")
    print("Default device:", jax.default_backend())
    print(
        "Note: HD correlation replaced by identity => common UNCORRELATED red noise.\n"
    )

    utils.check_gpu_availability()
    _install_curn_override()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = workflow.run_inference(
        config_path=args.config, use_gw=True, timestamp=timestamp
    )
    print(f"\nCURN inference complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
