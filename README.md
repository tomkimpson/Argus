# Argus

![Tests](https://github.com/tomkimpson/Argus/actions/workflows/run_test.yml/badge.svg) [![codecov](https://codecov.io/gh/tomkimpson/Argus/graph/badge.svg?token=2PEOHCFV1K)](https://codecov.io/gh/tomkimpson/Argus) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://tomkimpson.github.io/Argus/)

Welcome to **Argus**!

This package focuses on Bayesian inference for pulsar timing array data using JAX-accelerated Kalman filtering techniques.

Please see the [documentation](https://tomkimpson.github.io/Argus/).

## Data ingestion

Argus consumes lightweight per-pulsar Arrow **Feather** caches at runtime — it has **no
runtime dependency** on pulsar-timing parsers (no `enterprise`/`tempo2`). Parsing `.par`/
`.tim` files is a one-time, offline data-prep step:

```bash
# Run once in an environment that has enterprise installed (e.g. the `Argus` conda env):
python scripts/ingest_par_tim.py <par_tim_dir> <feather_out_dir>
```

This writes one `<pulsar>.feather` per pulsar. The analysis pipeline
(`LoadWidebandPulsarData.get_processed_residuals`) automatically prefers `*.feather` files
in a directory and falls back to `.par`/`.tim` only if no feathers are present. The cached
arrays are byte-identical to a direct `enterprise` load.

![Corner plot showing posterior distributions from Bayesian parameter estimation using Argus on the IPTA mock data challenge. The plot demonstrates the recovery of gravitational wave background parameters and pulsar noise characteristics using state-space Kalman filtering.](docs/joss/images/example_corner_plot.png)

