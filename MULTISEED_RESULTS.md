# Default Model Multi-Seed Results

## Protocol

- Model: ESM2 + geometry + learned-confidence fusion
- Training data: `data/processed/train_cached_func_esm2.pt`
- Seeds: 42, 123, 2025
- Validation: seed-dependent 15% training split
- Checkpoint selection: validation MCC
- Test threshold: fixed at 0.85 for every seed
- Test1 and test2 were not used for checkpoint or threshold selection
- Reported spread: sample standard deviation across the three runs

## Per-Seed Results

### Test1

| Seed | Accuracy | BACC | F1 | MCC | AUROC | AUPRC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9378 | 0.9179 | 0.8885 | 0.8458 | 0.9699 | 0.9426 |
| 123 | 0.9290 | 0.8986 | 0.8689 | 0.8222 | 0.9649 | 0.9276 |
| 2025 | 0.9165 | 0.8833 | 0.8459 | 0.7906 | 0.9659 | 0.9306 |
| Mean +/- SD | 0.9278 +/- 0.0107 | 0.8999 +/- 0.0174 | 0.8678 +/- 0.0213 | 0.8195 +/- 0.0276 | 0.9669 +/- 0.0027 | 0.9336 +/- 0.0079 |

### Test2

| Seed | Accuracy | BACC | Precision | Recall | F1 | MCC | AUROC | AUPRC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9399 | 0.8481 | 0.5965 | 0.7391 | 0.6602 | 0.6320 | 0.9316 | 0.6969 |
| 123 | 0.9347 | 0.7956 | 0.5800 | 0.6304 | 0.6042 | 0.5692 | 0.9238 | 0.6217 |
| 2025 | 0.9278 | 0.7820 | 0.5385 | 0.6087 | 0.5714 | 0.5334 | 0.9120 | 0.5882 |
| Mean +/- SD | 0.9341 +/- 0.0060 | 0.8086 +/- 0.0349 | 0.5717 +/- 0.0299 | 0.6594 +/- 0.0699 | 0.6119 +/- 0.0449 | 0.5782 +/- 0.0499 | 0.9225 +/- 0.0099 | 0.6356 +/- 0.0557 |

## Interpretation

The ranking metrics are comparatively stable on test1, but test2 decision metrics vary materially across random seeds. The seed-42 checkpoint remains the frozen default artifact because it was selected before this reproducibility analysis, but it should not be presented as the expected performance of a generic run. Paper tables should report the three-seed mean and standard deviation, with the seed-42 result optionally retained as the originally selected single-run result.

The variation is likely amplified by test2 containing only 46 positive samples. The next credibility step is bootstrap confidence intervals over fixed predictions, followed by train-test sequence-similarity analysis.

Bootstrap confidence intervals have now been completed and are reported in `BOOTSTRAP_RESULTS.md`.

## Artifacts

- Seed 42: `runs/plm_fusion_esm2_geometry_confidence/`
- Seed 123: `runs/plm_fusion_esm2_geometry_confidence_seed123/`
- Seed 2025: `runs/plm_fusion_esm2_geometry_confidence_seed2025/`
