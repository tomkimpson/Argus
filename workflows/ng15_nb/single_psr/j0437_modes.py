#!/usr/bin/env python
"""Discovery-OU lnL against its Fourier mode count at one (γp, σp) point (N3, J0437−4715).

C4's J0437−4715 outlier: Argus detects red noise at a fast corner (log10 γp ≈ −6.2,
log10 σp ≈ −13.4) where 30-mode Discovery-OU sees only an upper limit. The other
outlier was run the same way, with `--pulsar J1012+5307 --gamma-p -6.05`. This prints
Discovery's lnL relative to no red noise along σp at that γp, for 30, 100 and 300 modes on
the array span T. If the gain appears only with more modes, the power Argus finds lies
above 30/T, outside the 30-mode basis.

    conda activate discovery
    python workflows/ng15_nb/single_psr/j0437_modes.py [--pulsar J0437-4715]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_discovery as rd  # noqa: E402

MODES = (30, 100, 300)
SIGMA_P = (-16.0, -14.0, -13.4, -13.0)
NO_RN = (-8.0, -22.0)  # (log10 γp, log10 σp) at the σp floor


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pulsar", default="J0437-4715")
    ap.add_argument("--gamma-p", type=float, default=-6.2)
    args = ap.parse_args()
    psr = rd.load(args.pulsar)
    span_s = common.array_span_s(rd.DISCOVERY_DIR)
    for n in MODES:
        rd.RN_COMPONENTS = n
        loglike = rd.make_loglike(psr, span_s)
        base = float(loglike(*NO_RN))
        row = [round(float(loglike(args.gamma_p, s)) - base, 2) for s in SIGMA_P]
        print(
            f"{n} modes: lnL - lnL(no RN) at log10 σp {SIGMA_P}, "
            f"log10 γp {args.gamma_p}: {row}",
            flush=True,
        )


if __name__ == "__main__":
    main()
