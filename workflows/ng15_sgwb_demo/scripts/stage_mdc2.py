#!/usr/bin/env python
"""Stage per-pulsar MDC2 input directories for the M1 Stage A single-pulsar runs.

The Argus data loader globs a whole directory, and the only selection knob is
``excluded_psrs`` — so a single-pulsar run needs its own one-feather input
directory. For every ``<PSR>.feather`` in the MDC2 feather cache this script
creates::

    <output-dir>/<PSR>/<PSR>.feather   (symlink to the cache)
    <output-dir>/<PSR>/psr_noise.json  (that pulsar's efac/equad entry copied
                                        from group1_psr_noise.json)

The per-pulsar noise JSON exists because ``utils.get_efac_equad_injections``
loads *every* pulsar in the file it is given; a single-pulsar run must see a
single-entry file so the EFAC/EQUAD arrays have length 1.

Prerequisite (run once, in the Argus conda env with enterprise available)::

    JAX_PLATFORMS=cpu python scripts/ingest_par_tim.py \
        workflows/data/IPTA_MockDataChallenge2/dataset_2b \
        workflows/ng15_sgwb_demo/data/mdc2_all

Then::

    python workflows/ng15_sgwb_demo/scripts/stage_mdc2.py

The sorted pulsar list printed at the end is the canonical ordering used by
every downstream M1 artifact (Stage A medians pickle, empirical priors JSON).
"""

import argparse
import glob
import json
import os

_WORKFLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(_WORKFLOW_DIR))

DEFAULT_FEATHER_DIR = os.path.join(_WORKFLOW_DIR, "data", "mdc2_all")
DEFAULT_OUTPUT_DIR = os.path.join(_WORKFLOW_DIR, "data", "mdc2_singles")
DEFAULT_NOISE_JSON = os.path.join(
    _REPO_ROOT, "workflows", "data", "IPTA_MockDataChallenge2", "group1_psr_noise.json"
)


def stage(feather_dir, output_dir, noise_json_path, overwrite=False):
    """Build one input directory per pulsar from the MDC2 feather cache.

    Parameters
    ----------
    feather_dir : str
        Directory holding the ingested ``<PSR>.feather`` files.
    output_dir : str
        Parent directory for the per-pulsar subdirectories (created if absent).
    noise_json_path : str
        Path to the MDC2 truth white-noise JSON (group1_psr_noise.json).
    overwrite : bool, optional
        If True, replace existing symlinks/JSONs; otherwise error on collision.

    Returns
    -------
    list of str
        Sorted pulsar names staged.
    """
    feathers = sorted(glob.glob(os.path.join(feather_dir, "*.feather")))
    if not feathers:
        raise FileNotFoundError(
            f"No .feather files in {feather_dir} — run scripts/ingest_par_tim.py first "
            "(see module docstring)"
        )

    with open(noise_json_path, "r") as f:
        noise_params = json.load(f)

    psr_names = []
    for feather in feathers:
        psr = os.path.splitext(os.path.basename(feather))[0]
        if psr not in noise_params:
            raise KeyError(f"{psr} has a feather but no entry in {noise_json_path}")

        psr_dir = os.path.join(output_dir, psr)
        os.makedirs(psr_dir, exist_ok=True)

        link = os.path.join(psr_dir, os.path.basename(feather))
        if os.path.islink(link) or os.path.exists(link):
            if not overwrite:
                raise FileExistsError(f"{link} already exists (use --overwrite)")
            os.remove(link)
        os.symlink(os.path.abspath(feather), link)

        psr_noise_path = os.path.join(psr_dir, "psr_noise.json")
        if os.path.exists(psr_noise_path) and not overwrite:
            raise FileExistsError(f"{psr_noise_path} already exists (use --overwrite)")
        with open(psr_noise_path, "w") as f:
            json.dump({psr: noise_params[psr]}, f, indent=2)

        psr_names.append(psr)

    print(f"Staged {len(psr_names)} single-pulsar directories under {output_dir}:")
    for i, psr in enumerate(psr_names):
        print(f"  [{i:2d}] {psr}")
    return psr_names


def main():
    """Parse command-line arguments and stage the directories."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--feather-dir",
        default=DEFAULT_FEATHER_DIR,
        help="MDC2 feather cache directory (from ingest_par_tim.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory for the per-pulsar subdirectories",
    )
    parser.add_argument(
        "--noise-json",
        default=DEFAULT_NOISE_JSON,
        help="MDC2 truth white-noise JSON (group1_psr_noise.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing symlinks/JSONs instead of erroring",
    )
    args = parser.parse_args()

    stage(args.feather_dir, args.output_dir, args.noise_json, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
