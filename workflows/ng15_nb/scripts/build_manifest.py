"""Pin the NG15 narrowband input data: hash Discovery's feathers into MANIFEST.json.

Spec task 1.1. The NG15 narrowband data come from Discovery's bundled feathers
(github.com/nanograv/discovery, ``data/v1p1_de440_pint_bipm2019-*.feather``).
This script records the Discovery commit, the data-set version (parsed from the
feather filename prefix; ``v1p1`` = NG15 v1.1.0, the version in Agazie et al.
2023), the SHA-256 of every feather, and the pulsar name embedded in each.

Usage (any env with pyarrow, e.g. ``conda activate Argus``)::

    git clone https://github.com/nanograv/discovery.git <DISCOVERY_DIR>
    git -C <DISCOVERY_DIR> checkout <PINNED_COMMIT>
    python workflows/ng15_nb/scripts/build_manifest.py <DISCOVERY_DIR>

    # later, to check a checkout against the committed manifest:
    python workflows/ng15_nb/scripts/build_manifest.py <DISCOVERY_DIR> --check
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.feather as feather

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "data" / "ng15_nb" / "MANIFEST.json"
PREFIX_RE = re.compile(r"^(v(\d+)p(\d+))_(.+)-([BJ]\d{4}[+-]\d{2,4}[A-Z]?)\.feather$")
N_PULSARS = 67


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(discovery_dir):
    discovery_dir = Path(discovery_dir).resolve()
    commit = subprocess.check_output(["git", "-C", str(discovery_dir), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(discovery_dir), "status", "--porcelain", "data"], text=True)
    if dirty.strip():
        sys.exit(f"Discovery data/ has local modifications:\n{dirty}")

    feathers = sorted((discovery_dir / "data").glob("v*p*_*-*.feather"))
    entries, versions = [], set()
    for path in feathers:
        m = PREFIX_RE.match(path.name)
        if m is None:
            sys.exit(f"unexpected feather name: {path.name}")
        tag, major, minor, ephem, psr = m.groups()
        meta = json.loads(feather.read_table(path, columns=["toas"]).schema.metadata[b"json"])
        if meta["name"] != psr:
            sys.exit(f"{path.name}: embedded name {meta['name']} != filename {psr}")
        versions.add((tag, f"{major}.{minor}.0", ephem))
        entries.append({"pulsar": psr, "file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})

    if len(entries) != N_PULSARS:
        sys.exit(f"expected {N_PULSARS} feathers, found {len(entries)}")
    if len(versions) != 1:
        sys.exit(f"mixed data-set versions: {versions}")
    (tag, version, ephem), = versions

    return {
        "source": "https://github.com/nanograv/discovery",
        "commit": commit,
        "dataset": "NANOGrav 15-yr narrowband",
        "dataset_version": version,
        "dataset_tag": f"{tag}_{ephem}",
        "n_pulsars": len(entries),
        "pulsars": [e["pulsar"] for e in entries],
        "feathers": entries,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("discovery_dir")
    ap.add_argument("--check", action="store_true", help="compare against the committed manifest; write nothing")
    args = ap.parse_args()

    manifest = build(args.discovery_dir)
    if args.check:
        committed = json.loads(MANIFEST.read_text())
        if committed != manifest:
            sys.exit("MISMATCH: Discovery checkout differs from committed MANIFEST.json")
        print(f"OK: {manifest['n_pulsars']} feathers match {MANIFEST.relative_to(REPO_ROOT)}")
        return

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}: commit {manifest['commit'][:10]}, "
          f"{manifest['dataset']} v{manifest['dataset_version']} ({manifest['dataset_tag']}), "
          f"{manifest['n_pulsars']} feathers hashed")


if __name__ == "__main__":
    main()
