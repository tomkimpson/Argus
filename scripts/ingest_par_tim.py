#!/usr/bin/env python
"""Offline ingestion: convert pulsar .par/.tim files into Argus feather caches.

This is a *data-prep* tool, not part of the Argus runtime. It is the only place
that imports ``enterprise`` (via ``LoadWidebandPulsarData.read_par_tim``), so it
must be run in an environment where enterprise/tempo2 are installed (the ``Argus``
conda env). It parses each ``.par``/``.tim`` pair once and writes one
``<pulsar>.feather`` per pulsar. The Argus runtime (and CI) then load those
feathers via ``LoadWidebandPulsarData.read_feather`` with no enterprise dependency.

Example
-------
    python scripts/ingest_par_tim.py \
        workflows/data/IPTA_MockDataChallenge2/dataset_3b \
        workflows/data/IPTA_MockDataChallenge2/dataset_3b
"""

import argparse
import glob
import os

from argus.data_loader import LoadWidebandPulsarData


def find_par_tim_pairs(input_dir):
    """Find matched ``.par``/``.tim`` pairs in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing ``.par`` and ``.tim`` files.

    Returns
    -------
    list of tuple of str
        ``(par_file, tim_file)`` pairs, sorted by par-file name.

    Raises
    ------
    ValueError
        If the number of ``.par`` and ``.tim`` files does not match.
    """
    par_files = sorted(glob.glob(os.path.join(input_dir, "*.par")))
    tim_files = sorted(glob.glob(os.path.join(input_dir, "*.tim")))

    if len(par_files) != len(tim_files):
        raise ValueError(
            f"Mismatch between .par ({len(par_files)}) and .tim "
            f"({len(tim_files)}) file counts in {input_dir}"
        )

    return list(zip(par_files, tim_files))


def ingest(input_dir, output_dir, excluded_psrs=(), max_files=None, overwrite=False):
    """Convert all ``.par``/``.tim`` pairs in ``input_dir`` to feather caches.

    Parameters
    ----------
    input_dir : str
        Directory containing ``.par``/``.tim`` files.
    output_dir : str
        Directory to write ``<pulsar>.feather`` files into (created if absent).
    excluded_psrs : sequence of str, optional
        Substrings of par-file paths to skip (e.g. pulsar names).
    max_files : int, optional
        If provided, only the first ``max_files`` pairs are processed.
    overwrite : bool, optional
        If False (default), skip pairs whose output feather already exists.

    Returns
    -------
    list of str
        Paths of the feather files written.
    """
    os.makedirs(output_dir, exist_ok=True)

    pairs = find_par_tim_pairs(input_dir)
    pairs = [p for p in pairs if not any(psr in p[0] for psr in excluded_psrs)]
    if max_files is not None:
        pairs = pairs[:max_files]

    written = []
    for i, (par_file, tim_file) in enumerate(pairs):
        try:
            psr = LoadWidebandPulsarData.read_par_tim(par_file, tim_file)
            f0 = LoadWidebandPulsarData.get_par_value(par_file, "F0")
            out_path = os.path.join(output_dir, f"{psr.name}.feather")

            if os.path.exists(out_path) and not overwrite:
                print(f"[{i + 1}/{len(pairs)}] skip (exists): {out_path}")
                written.append(out_path)
                continue

            psr.save_feather(out_path, F0=f0)
            print(
                f"[{i + 1}/{len(pairs)}] {psr.name}: "
                f"{len(psr.toas)} TOAs, F0={f0} -> {out_path}"
            )
            written.append(out_path)
        except Exception as e:  # noqa: BLE001 - report and continue per pulsar
            print(f"[{i + 1}/{len(pairs)}] ERROR on {par_file}: {e}")

    print(f"Done. Wrote {len(written)} feather files to {output_dir}")
    return written


def main():
    """Parse command-line arguments and run the ingestion."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", help="Directory with .par/.tim files")
    parser.add_argument("output_dir", help="Directory to write .feather files")
    parser.add_argument(
        "--excluded-psrs",
        nargs="*",
        default=[],
        help="Pulsar-name substrings to skip (e.g. J1640+2224)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only process the first N par/tim pairs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feather files instead of skipping them",
    )
    args = parser.parse_args()

    ingest(
        args.input_dir,
        args.output_dir,
        excluded_psrs=args.excluded_psrs,
        max_files=args.max_files,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
