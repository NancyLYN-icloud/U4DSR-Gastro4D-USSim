# Method

## U4DSR

U4DSR combines:

1. **PAOC** — nonlinear phase canonicalization and confidence-weighted phase-set construction;
2. **GBR** — structured dynamic modeling on observation-initialized shared topology.

```text
monitor stream  ──┐
                  ├──> PAOC ──> P_phi ──> GBR ──> V(phi)
scanner stream  ──┘
```

## Code map

| Module | Files |
| --- | --- |
| PAOC | `gbr_4d_ums/phase.py`, `gbr_4d_ums/observations.py` |
| GBR | `gbr_4d_ums/model.py`, `gbr_4d_ums/pipeline.py`, `gbr_4d_ums/topology.py` |

## Demo

`scripts/run_demo.py` runs PAOC and GBR on synthetic periodic observations and writes `demo/outputs/demo_summary.json`.

The full benchmark harness, mesh export, and complete loss terms are described in the manuscript supplementary material.
