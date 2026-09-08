#!/usr/bin/env python
"""Write a one-line provenance stamp into a run's output directory.

Why this exists
---------------
`sgwb/model-selection` requires that the evidence procedure be recorded as a fixed,
versioned configuration and that *every run record which frozen version it used*
(openspec sgwb-detection-route task 1.10). Without a stamp, a run's outputs cannot
be attributed to a procedure version after the fact, so results produced under
different versions cannot be kept apart -- which is exactly what the spec's
"procedure changed after freezing" scenario asks us to be able to do.

Why it is a separate script rather than a flag on the estimator
---------------------------------------------------------------
`lnb_path_sampling.py` IS the frozen procedure. Editing it -- even to add a stamp --
would change the artefact the freeze is defined against and mark every result
obtained under it stale. So the stamp is written alongside the run instead, by the
SLURM driver, after sampling completes.

Usage (from workflows/ng15_sgwb_demo):
    python scripts/stamp_provenance.py \
        --output-dir outputs/mdc2_d1_nogwb_eps000 \
        --config configs/mdc2_d1_nogwb_eps000.ini \
        --frozen configs/frozen/evidence_procedure_v1.json
"""

import argparse
import configparser
import datetime
import hashlib
import json
import os
import socket
import subprocess


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_sha(repo_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", required=True, help="The run's output directory")
    p.add_argument("--config", required=True, help="The rung config the run used")
    p.add_argument(
        "--frozen",
        default="configs/frozen/evidence_procedure_v1.json",
        help="The frozen evidence-procedure record (default: %(default)s)",
    )
    p.add_argument(
        "--repo",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        help="Repo root, for the git SHA",
    )
    args = p.parse_args()

    if not os.path.isdir(args.output_dir):
        raise SystemExit(f"*** {args.output_dir} is not a directory. ***")
    for path in (args.config, args.frozen):
        if not os.path.isfile(path):
            raise SystemExit(f"*** {path} not found. ***")

    with open(args.frozen) as f:
        frozen = json.load(f)

    # Read back the two config values that decide whether a run is comparable with
    # others under the same frozen version, so the stamp is self-contained.
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    def _get(section, key):
        try:
            return cfg.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return None

    stamp = {
        "frozen_procedure_version": frozen.get("version"),
        "frozen_procedure_file": os.path.relpath(args.frozen),
        "frozen_procedure_sha256": _sha256(args.frozen),
        "estimator": frozen.get("estimator", {}).get("name"),
        "git_sha": _git_sha(args.repo),
        "config": os.path.relpath(args.config),
        "config_sha256": _sha256(args.config),
        "data_path": _get("Data", "data_path"),
        "orf_epsilon_value": _get("PriorModel", "orf_epsilon_value"),
        "red_noise_prior": _get("PriorModel", "red_noise_prior"),
        "empirical_priors_path": _get("PriorModel", "empirical_priors_path"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "stamped_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }

    out = os.path.join(args.output_dir, "provenance.json")
    with open(out, "w") as f:
        json.dump(stamp, f, indent=2)
        f.write("\n")
    print(
        f"provenance: {out} -- frozen procedure {stamp['frozen_procedure_version']} "
        f"({stamp['estimator']}), git {(stamp['git_sha'] or 'unknown')[:8]}"
    )


if __name__ == "__main__":
    main()
