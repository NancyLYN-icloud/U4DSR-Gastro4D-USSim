# Gastro4D-USSim

Frozen benchmark for gastric freehand ultrasound 4D dynamic surface reconstruction.

## Scale

| Component | Specification | Scale |
| --- | --- | --- |
| Instances | MedPointS + 4D motion + dual-stream US | 281 |
| Morphology | HT, J, Steerhorn, Cascade | 4 |
| Protocols | Clean / Sparse / PoseNoise / ImageNoise | 4 |
| Cases | instance × protocol | 1,124 |
| Split | `dev` / `test` | 81 / 200 |

## Files

- `dataset_summary.json` — benchmark summary
- `manifests/manifest_schema.md` — manifest columns
- `manifests/example_rows.csv` — example entries

Simulation data and the frozen manifest are distributed under the data access procedure in `DATA_ACCESS.md`.

## Dataset request

See [`DATA_ACCESS.md`](DATA_ACCESS.md) and [`../docs/Gastro4D-USSim_Data_Access_Agreement.pdf`](../docs/Gastro4D-USSim_Data_Access_Agreement.pdf).

## Protocol

All methods in the paper share the same frozen phase-set interface `P_phi` from PAOC. This repository contains the GBR backend only.
