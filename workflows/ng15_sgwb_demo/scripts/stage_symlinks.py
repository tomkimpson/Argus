#!/usr/bin/env python
"""Stage co-located .par/.tim symlinks for the NG15 wideband pulsar subset.

The NG15 wideband release keeps ``.par`` and ``.tim`` in *separate* directories
(``par/`` and ``tim/``), but ``scripts/ingest_par_tim.py`` globs a *single*
directory for both. This script bridges that gap: for a chosen subset of pulsars
it creates symlinks to the canonical ``.par`` and ``.tim`` files inside one
staging directory, ready to hand to the ingester.

Two NG15 wrinkles are handled here:

1. **Telescope-suffixed duplicates.** Some pulsars have ``...ao`` / ``...gbt``
   single-telescope variants (e.g. ``B1937+21ao_PINT_*``). The canonical file has
   ``_PINT`` immediately after the pulsar name, so globbing ``{PSR}_PINT_*`` picks
   only the canonical file and skips the suffixed ones.

2. **Mismatched par/tim dates.** A pulsar's ``.par`` and ``.tim`` may carry
   different date stamps (e.g. J1600-3053 par ``...20230202`` vs tim
   ``...20230224``). ``ingest_par_tim.py`` pairs files by *sort position*, so we
   name the symlinks canonically (``{PSR}.wb.par`` / ``{PSR}.wb.tim``) to guarantee
   par and tim sort into the same order regardless of their source date stamps.

Example
-------
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/stage_symlinks.py

    # then ingest:
    JAX_PLATFORMS=cpu python scripts/ingest_par_tim.py \
        workflows/ng15_sgwb_demo/data/staging_subset \
        workflows/ng15_sgwb_demo/data
"""

import argparse
import glob
import os

# NG15 wideband release root (contains par/ and tim/ subdirectories).
DEFAULT_NG15_WIDEBAND = (
    "/fred/oz022/tkimpson/pta-solar-wind-2/data/"
    "NANOGrav15yr_PulsarTiming_v2.1.0/wideband"
)

# Science-motivated subset that dominates the NG15 HD S/N while keeping the
# epoch-alignment problem small (PLAN §4). Note NG15 names this pulsar
# B1855+09 (a.k.a. J1857+0943). J0437-4715 is deliberately excluded (PPTA, not
# in NG15). Order roughly by precision / baseline.
DEFAULT_SUBSET = [
    "J1909-3744",
    "J1713+0747",
    "J1744-1134",
    "J0613-0200",
    "B1855+09",
    "J1600-3053",
]


def enumerate_canonical_pulsars(par_dir):
    """Return the sorted list of canonical NG15 pulsar names in ``par_dir``.

    A canonical file is ``{PSR}_PINT_YYYYMMDD.wb.par`` with ``_PINT`` immediately
    after the pulsar name; single-telescope variants have a ``...ao`` / ``...gbt``
    suffix on the name (e.g. ``B1937+21ao_PINT_*``) and are dropped so only the
    combined-telescope solution is kept. NG15 has 68 such pulsars.

    Parameters
    ----------
    par_dir : str
        The release's ``par/`` directory.

    Returns
    -------
    list of str
        Sorted canonical pulsar names.
    """
    names = set()
    for path in glob.glob(os.path.join(par_dir, "*_PINT_*.wb.par")):
        base = os.path.basename(path)
        name = base.split("_PINT_", 1)[0]
        if name.endswith("ao") or name.endswith("gbt"):
            continue
        names.add(name)
    return sorted(names)


def _find_canonical(directory, psr, ext):
    """Return the single canonical ``{psr}_PINT_*.{ext}`` file in ``directory``.

    Parameters
    ----------
    directory : str
        Directory to search (the release's ``par/`` or ``tim/``).
    psr : str
        Pulsar name (e.g. ``J1909-3744``).
    ext : str
        File extension without the dot: ``"par"`` or ``"tim"``.

    Returns
    -------
    str
        Path to the single matching file.

    Raises
    ------
    FileNotFoundError
        If no canonical file matches.
    ValueError
        If more than one canonical file matches (ambiguous).
    """
    pattern = os.path.join(directory, f"{psr}_PINT_*.wb.{ext}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No canonical .{ext} for {psr} (pattern: {pattern})")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous canonical .{ext} for {psr}: {[os.path.basename(m) for m in matches]}"
        )
    return matches[0]


def stage(ng15_wideband, output_dir, subset=DEFAULT_SUBSET, overwrite=False):
    """Symlink canonical .par/.tim for ``subset`` into one staging directory.

    Parameters
    ----------
    ng15_wideband : str
        Path to the NG15 wideband release root (holds ``par/`` and ``tim/``).
    output_dir : str
        Staging directory to populate with symlinks (created if absent).
    subset : sequence of str, optional
        Pulsar names to stage. Defaults to :data:`DEFAULT_SUBSET`.
    overwrite : bool, optional
        If True, replace existing symlinks; otherwise error on collision.

    Returns
    -------
    list of tuple of str
        ``(par_link, tim_link)`` staged symlink paths, one per pulsar.
    """
    par_dir = os.path.join(ng15_wideband, "par")
    tim_dir = os.path.join(ng15_wideband, "tim")
    for d in (par_dir, tim_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected NG15 subdirectory not found: {d}")

    os.makedirs(output_dir, exist_ok=True)

    staged = []
    for psr in subset:
        par_src = _find_canonical(par_dir, psr, "par")
        tim_src = _find_canonical(tim_dir, psr, "tim")

        # Canonical link names guarantee par/tim sort into matching order for
        # ingest_par_tim.py's positional pairing, even when source date stamps differ.
        par_link = os.path.join(output_dir, f"{psr}.wb.par")
        tim_link = os.path.join(output_dir, f"{psr}.wb.tim")

        for link, src in ((par_link, par_src), (tim_link, tim_src)):
            if os.path.islink(link) or os.path.exists(link):
                if not overwrite:
                    raise FileExistsError(
                        f"{link} already exists (use --overwrite to replace)"
                    )
                os.remove(link)
            os.symlink(os.path.abspath(src), link)

        staged.append((par_link, tim_link))
        print(
            f"{psr}: "
            f"{os.path.basename(par_src)} + {os.path.basename(tim_src)} "
            f"-> {os.path.basename(par_link)} / {os.path.basename(tim_link)}"
        )

    print(
        f"Done. Staged {len(staged)} pulsars "
        f"({len(staged)} .par + {len(staged)} .tim symlinks) into {output_dir}"
    )
    return staged


def main():
    """Parse command-line arguments and stage the symlinks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ng15-wideband",
        default=DEFAULT_NG15_WIDEBAND,
        help="NG15 wideband release root (contains par/ and tim/)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "staging_subset",
        ),
        help="Staging directory to populate with symlinks",
    )
    parser.add_argument(
        "--subset",
        nargs="*",
        default=DEFAULT_SUBSET,
        help="Pulsar names to stage (default: the science subset)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Stage the full canonical array (all ~68 NG15 pulsars, ao/gbt variants "
            "excluded), overriding --subset. For the full-array (T3.5) run."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing symlinks instead of erroring",
    )
    args = parser.parse_args()

    if args.all:
        par_dir = os.path.join(args.ng15_wideband, "par")
        subset = enumerate_canonical_pulsars(par_dir)
        print(f"--all: staging the full canonical array ({len(subset)} pulsars)")
    else:
        subset = args.subset

    stage(
        args.ng15_wideband,
        args.output_dir,
        subset=subset,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
