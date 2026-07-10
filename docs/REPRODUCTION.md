# Reproduction

## Contents

- U4DSR method code (PAOC + GBR)
- Gastro4D-USSim benchmark metadata and manifest schema
- Synthetic demo (`scripts/run_demo.py`)

Simulation volumes, frozen manifests, per-instance metrics, and full reproduction scripts are released with the **Gastro4D-USSim** archive. Request access via [`benchmark/DATA_ACCESS.md`](../benchmark/DATA_ACCESS.md).

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_demo.py
```

## Benchmark protocol (paper)

| Setting | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| Dynamic train steps | 15,000 |
| Mesh resolution | 72 |
| Max points per phase | 3,000 |

Equal-budget protocol: `historical_best_eqbudget`. Main-text results use the `test` split (200 instances, 800 cases).

## Data layout

```text
$GASTRO4D_USSIM_ROOT/
  benchmark/manifests/benchmark_condition_manifest.csv
  benchmark/instances/<instance_name>/...
  benchmark/conditions/<protocol>/instances/<instance_name>/...
  results/metrics/...
```

Set `GASTRO4D_USSIM_ROOT` before running the full harness.
