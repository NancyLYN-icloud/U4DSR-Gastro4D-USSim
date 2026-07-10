# Manifest schema

## benchmark_condition_manifest.csv

| Column | Description |
| --- | --- |
| `instance_name` | Instance identifier |
| `morphology_type` | horn-type / J-shaped / steerhorn / cascade |
| `split` | dev or test |
| `condition` | Clean / Sparse / PoseNoise / ImageNoise |
| `monitor_stream` | Monitor stream (`*.npz`) |
| `scanner_sequence` | Scanner stream (`*.npz`) |
| `phase_model_dir` | Phase point-cloud directory |
| `phase_summary` | Phase summary CSV |
| `condition_root` | Case root directory |
| `condition_metadata` | Protocol metadata JSON |

## benchmark_instance_manifest.csv

| Column | Description |
| --- | --- |
| `instance_name` | Instance identifier |
| `morphology_type` | Morphology label |
| `split` | dev or test |
| `reference_mesh` | Reference mesh for topology initialization |
| `monitor_stream` | Clean-protocol monitor stream |
| `scanner_sequence` | Clean-protocol scanner stream |

Set `GASTRO4D_USSIM_ROOT` to the benchmark archive root before running the harness.
