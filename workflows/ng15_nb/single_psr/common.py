"""Shared pieces of the single-pulsar noise runs (spec Phase 3, check C4).

Both pipelines, Argus (`run_argus.py`, Argus env) and Discovery-OU (`run_discovery.py`,
discovery env), sample the same two sites per pulsar under the same exact-uniform boxes
and reduce their draws through this module, so their JSON summaries are comparable field
by field. Numpy, arviz and pyarrow only: it must import in both environments.

Sites, in Argus's own basis (as in `reference/configs/ou.ini`):
- ``log10_gamma_p``: the spin-noise OU corner γp [rad/s];
- ``log10_sigma_p``: the spin-noise driving amplitude σp [spin units]; the residual sees
  σp / f0.

The model is red noise only, with no common process: these are single-pulsar noise runs.

What the summaries record (task 3.1's verify clause):
- r̂ (rank-normalised) and bulk ESS per site, and the divergence count;
- the posterior mass near each box edge. A pulsar *without* detected red noise piles up
  at the lower σp edge, which is expected and reported. A pulsar *with* detected red noise
  must not rail against the σp edges. Neither γp edge is gated (spec §12, 2026-10-20):
  - the floor: a corner below the band (ω ≪ 2π/T) leaves an f⁻⁴ spectrum with γp
    unconstrained, so mass there is the prior;
  - the ceiling: a corner above the band leaves σ²/(γ²ω²), an f⁻² spectrum that depends
    on σp/γp alone, so the likelihood is flat along γp → ∞ and widening the box only
    moves the pile-up. Mass there means the data want OU's shallowest spectrum or
    shallower (A3's shallow-spectrum exposure), and is flagged ``at_f2_limit``;
- the one-sided spin-noise residual PSD at f = 1/T (T the 67-pulsar array span) and
  f = 1/(5 yr), as log10 s²/Hz quantiles. Those are C4's comparison points.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "workflows/ng15_nb/reference"))
from ou_psd import spin_psd_onesided  # noqa: E402

MANIFEST = REPO_ROOT / "data/ng15_nb/MANIFEST.json"
F0_CATALOG = REPO_ROOT / "data/ng15_nb/f0_catalog.json"
EPOCH_RULE = REPO_ROOT / "workflows/ng15_nb/results/c1/epoch_rule.json"
RESULTS = REPO_ROOT / "workflows/ng15_nb/results/single_psr"
OUTPUTS = REPO_ROOT / "workflows/ng15_nb/outputs/single_psr"

YR_S = 365.25 * 86400.0
# Exact uniform boxes, identical to reference/configs/ou.ini's red-noise sites.
BOX = {"log10_gamma_p": (-12.0, -6.0), "log10_sigma_p": (-22.0, -11.5)}
SITES = tuple(BOX)

# NUTS block of evidence_procedure_v1.json, with 4000 draws per chain instead of 1000:
# the γp floor is a flat (f⁻⁴) direction that mixes slowly, and a single-pulsar draw
# costs little, so the longer run is what makes task 3.1's ESS ≥ 400 reachable.
NUTS = {
    "num_chains": 4,
    "num_warmup": 1000,
    "num_samples": 4000,
    "target_accept_prob": 0.9,
    "max_tree_depth": 8,
}
SEED_BASE = 3100  # seed = SEED_BASE + the pulsar's manifest index

# Task 3.1's thresholds.
RHAT_MAX = 1.01
ESS_MIN = 400
EDGE_WIDTH = 0.02  # an "edge" is the outer 2% of a box
EDGE_MASS_RAIL = (
    0.05  # a detected pulsar rails if > 5% of its draws sit at a gated edge
)
EDGE_MASS_DETECT = 0.01  # red noise is "detected" if < 1% of σp draws sit at its floor

# The five shallow-spectrum pulsars (γ_PL ≈ 0.2-1.1) where the OU kernel fails its
# wideband adequacy test (workflows/ng15_sgwb_demo/notes/RESULTS_ou_adequacy_ng15.md).
SHALLOW = ("J1012+5307", "J1643-1224", "J1705-1903", "J1903+0327", "J2234+0944")


def pulsars():
    """The 67 pulsar names in manifest order (the SLURM array index)."""
    return [e["pulsar"] for e in json.loads(MANIFEST.read_text())["feathers"]]


def f0_of(name):
    return float(json.loads(F0_CATALOG.read_text())["f0_hz"][name])


def uses_epoch_cells(name):
    """D2's rule (spec §4): the pulsars whose window compression misses C1's spread."""
    return name in json.loads(EPOCH_RULE.read_text())["needs_epoch_cells"]


def seed_of(name):
    return SEED_BASE + pulsars().index(name)


def array_span_s(discovery_dir):
    """T: Discovery's getspan over the 67 pinned feathers (N2 quotes 16.030 yr)."""
    from kernel_systematic import array_span_s as span

    return span(discovery_dir)


def frequencies(span_s):
    return {"1/T": 1.0 / span_s, "1/(5 yr)": 1.0 / (5.0 * YR_S)}


def git_sha(repo=REPO_ROOT):
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def edge_fractions(x, lo, hi):
    """Fraction of draws in the bottom and top EDGE_WIDTH of the box [lo, hi]."""
    w = EDGE_WIDTH * (hi - lo)
    return float(np.mean(x < lo + w)), float(np.mean(x > hi - w))


def edges(samples, box=None):
    """Per-site edge masses, red-noise detection, and the rail verdict (see module doc)."""
    out = {}
    for s, (lo, hi) in (box or BOX).items():
        f_lo, f_hi = edge_fractions(np.asarray(samples[s]).ravel(), lo, hi)
        out[s] = {"floor": f_lo, "ceiling": f_hi}
    return verdict(out)


def verdict(edge_mass):
    """Detection, rails and the f⁻² flag from the per-site edge masses."""
    detected = edge_mass["log10_sigma_p"]["floor"] < EDGE_MASS_DETECT
    gated = {
        "log10_sigma_p floor": edge_mass["log10_sigma_p"]["floor"],
        "log10_sigma_p ceiling": edge_mass["log10_sigma_p"]["ceiling"],
    }
    rails = [k for k, v in gated.items() if v > EDGE_MASS_RAIL] if detected else []
    return {
        "edge_mass": edge_mass,
        "red_noise_detected": bool(detected),
        "lower_edge_pileup": not detected,
        "at_f2_limit": bool(
            detected and edge_mass["log10_gamma_p"]["ceiling"] > EDGE_MASS_RAIL
        ),
        "rails": rails,
    }


def passes_3_1(diag, edge):
    """Task 3.1's verify clause: r̂, bulk ESS, no divergences, no rails."""
    return bool(
        diag["max_r_hat"] <= RHAT_MAX
        and diag["min_ess_bulk"] >= ESS_MIN
        and diag["divergences"] == 0
        and not edge["rails"]
    )


def diagnostics(samples, diverging):
    """r̂ and bulk ESS per site (arviz, rank-normalised), and the divergence count.

    ``samples``: {site: (chains, draws)}; ``diverging``: (chains, draws) bools.
    """
    import arviz as az

    ds = az.convert_to_dataset({s: np.asarray(samples[s]) for s in SITES})
    rhat = az.rhat(ds)
    ess = az.ess(ds, method="bulk")
    per_site = {s: {"r_hat": float(rhat[s]), "ess_bulk": float(ess[s])} for s in SITES}
    n_div = int(np.sum(diverging))
    return {
        "per_site": per_site,
        "max_r_hat": max(v["r_hat"] for v in per_site.values()),
        "min_ess_bulk": min(v["ess_bulk"] for v in per_site.values()),
        "divergences": n_div,
        "divergence_fraction": float(np.mean(diverging)),
    }


def log10_psd_draws(samples, f0, f):
    """One-sided spin-noise residual PSD at f for every draw, log10 s²/Hz."""
    return np.log10(
        spin_psd_onesided(
            f,
            np.asarray(samples["log10_gamma_p"]).ravel(),
            np.asarray(samples["log10_sigma_p"]).ravel(),
            f0,
        )
    )


def quantiles(x):
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return {"median": float(q50), "q16": float(q16), "q84": float(q84)}


def summarise(name, pipeline, samples, diverging, span_s, extra=None, box=None):
    """The per-pulsar JSON record both pipelines write."""
    f0 = f0_of(name)
    diag = diagnostics(samples, diverging)
    box = box or BOX
    edge = edges(samples, box)
    psd = {
        label: quantiles(log10_psd_draws(samples, f0, f))
        for label, f in frequencies(span_s).items()
    }
    passes = passes_3_1(diag, edge)
    rec = {
        "pulsar": name,
        "pipeline": pipeline,
        "f0_hz": f0,
        "shallow_spectrum": name in SHALLOW,
        "T_yr": span_s / YR_S,
        "freqs_hz": frequencies(span_s),
        "box": {k: list(v) for k, v in box.items()},
        "nuts": dict(NUTS, seed=seed_of(name)),
        "diagnostics": diag,
        **edge,
        "passes_3_1": bool(passes),
        "site_quantiles": {s: quantiles(np.asarray(samples[s]).ravel()) for s in SITES},
        "log10_psd": psd,
        "git_sha": git_sha(),
    }
    rec.update(extra or {})
    return rec


def write(rec):
    out = RESULTS / rec["pipeline"] / f"{rec['pulsar']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    return out
