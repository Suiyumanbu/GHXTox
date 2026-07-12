# GHXTox Experiment Log

## Reproducibility Scope

This log records the frozen default model and the experiments used in the final result tables. The working tree contains uncommitted research changes; Git commit `7e47152` is therefore a baseline reference, not a complete experiment snapshot.

### Environment

| Item | Value |
| --- | --- |
| Operating system | Windows local training environment |
| Python executable | `D:/anaconda3/envs/dachuang-26/python.exe` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch | 2.7.1+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| NumPy | 2.2.6 |
| scikit-learn | 1.7.2 |
| Matplotlib | 3.10.9 |

All module commands require:

```powershell
$env:PYTHONPATH="src"
$py="D:\anaconda3\envs\dachuang-26\python.exe"
```

## Data

| Split | FASTA | Processed ESM2/structure cache | N | Positive | Negative |
| --- | --- | --- | ---: | ---: | ---: |
| Train | `dataset/train_data or benchmark_data.fasta` | `data/processed/train_cached_func_esm2.pt` | 6387 | 1818 | 4569 |
| Test1 | `dataset/test1.fasta` | `data/processed/test1_cached_func_esm2.pt` | 1126 | 320 | 806 |
| Test2 | `dataset/test2.fasta` | `data/processed/test2_cached_func_esm2.pt` | 582 | 46 | 536 |

PLM features are residue-level ESM2-650M embeddings with dimension 1280. The primary structure representation uses ESMFold C-alpha coordinates, pLDDT, and residue geometry descriptors.

## Frozen Default Run

| Item | Value |
| --- | --- |
| Config | `configs/default.json` |
| Config SHA256 | `82b390c4f4ee6577457f056dc9fc080c87e81beb677d77bea09707ef9d147664` |
| Checkpoint | `runs/plm_fusion_esm2_geometry_confidence/best_model.pt` |
| Checkpoint SHA256 | `9d99d37752d6c433b42382c60cbb1ed9ee5d62a8a5962d45ed8d275988c765c4` |
| Seed | 42 |
| Best epoch | 15 |
| Validation monitor | MCC |
| Best validation MCC | 0.866369 |
| Fixed decision threshold | 0.85 |

Training command:

```powershell
& $py -B -m ghxtox.train `
  --train data\processed\train_cached_func_esm2.pt `
  --config configs\default.json `
  --output-dir runs\plm_fusion_esm2_geometry_confidence
```

Evaluation commands:

```powershell
& $py -B -m ghxtox.evaluate `
  --checkpoint runs\plm_fusion_esm2_geometry_confidence\best_model.pt `
  --processed data\processed\test1_cached_func_esm2.pt `
  --output runs\plm_fusion_esm2_geometry_confidence\test1_metrics.json `
  --predictions runs\plm_fusion_esm2_geometry_confidence\test1_predictions.csv `
  --threshold 0.85

& $py -B -m ghxtox.evaluate `
  --checkpoint runs\plm_fusion_esm2_geometry_confidence\best_model.pt `
  --processed data\processed\test2_cached_func_esm2.pt `
  --output runs\plm_fusion_esm2_geometry_confidence\test2_metrics.json `
  --predictions runs\plm_fusion_esm2_geometry_confidence\test2_predictions.csv `
  --threshold 0.85
```

The prediction CSV writer uses nine significant digits so float32 probabilities round-trip without changing ranking metrics.

## Multi-Seed Runs

| Seed | Config | Output directory | Best epoch | Validation MCC |
| ---: | --- | --- | ---: | ---: |
| 42 | `configs/plm_fusion_esm2_geometry_confidence.json` | `runs/plm_fusion_esm2_geometry_confidence` | 15 | 0.866369 |
| 123 | `configs/plm_fusion_esm2_geometry_confidence_seed123.json` | `runs/plm_fusion_esm2_geometry_confidence_seed123` | 17 | 0.881657 |
| 2025 | `configs/plm_fusion_esm2_geometry_confidence_seed2025.json` | `runs/plm_fusion_esm2_geometry_confidence_seed2025` | 10 | 0.872867 |

The three configs are structurally identical after removing the `seed` field. Every test evaluation uses threshold 0.85.

## Bootstrap Analysis

Command pattern:

```powershell
& $py -B -m ghxtox.bootstrap_ci `
  --predictions <seed42.csv> <seed123.csv> <seed2025.csv> `
  --output runs\bootstrap_ci\<test>_multiseed.json `
  --threshold 0.85 `
  --iterations 5000 `
  --confidence 0.95 `
  --seed 2026
```

Protocol: class-stratified percentile bootstrap for individual runs and hierarchical seed-then-sample bootstrap for aggregate intervals. Machine-readable outputs are in `runs/bootstrap_ci/`.

## Structure-Quality Experiment

- Filter: mean pLDDT >= 0.70
- Filtered train/test1/test2 sizes: 3726/671/333
- Training config: `configs/plm_fusion_esm2_geometry_confidence_plddt070.json`
- Output: `runs/plm_fusion_esm2_geometry_confidence_plddt070/`
- Conclusion: high-quality evaluation subsets perform better, but hard-filtered training performs worse on full tests.

## Full-Backbone Geometry Experiment

Five coordinate slots were cached per residue: N, CA, C, O, and side-chain centroid. N-CA-C frames and corresponding-atom pair geometry were added to spatial edges.

- Config: `configs/plm_fusion_esm2_full_backbone_geometry_confidence.json`
- Training cache: `data/processed/train_full_backbone_esm2.pt`
- Output: `runs/plm_fusion_esm2_full_backbone_geometry_confidence/`
- Best epoch: 5
- Conclusion: test2 MCC 0.5497 and F1 0.5814; rejected.

## ProtT5 Control

- Local model: `models/prot_t5_xl_half_uniref50-enc`
- Embedding dimension: 1024
- Config: `configs/plm_fusion_prott5_geometry_confidence.json`
- Output: `runs/plm_fusion_prott5_geometry_confidence/`
- Best epoch: 19
- Conclusion: ranking is competitive, but test2 MCC/F1 are below ESM2.

## CD-HIT Experiment

CD-HIT 4.8.1 was run on a Linux server. Combined train+test clustering was required because the initial `cd-hit-2d` outputs showed directional length misses.

| Threshold | Train representatives | Test1 strict | Test2 strict |
| ---: | ---: | ---: | ---: |
| 0.90 | 6380 | 1123 | 580 |
| 0.80 | 5183 | 771 | 425 |

CD-HIT 0.80 retraining:

```powershell
& $py -B -m ghxtox.train `
  --train data\processed\train_cached_func_esm2_cdhit080.pt `
  --config configs\plm_fusion_esm2_geometry_confidence.json `
  --output-dir runs\plm_fusion_esm2_geometry_confidence_cdhit080
```

- Best epoch: 14
- Validation MCC: 0.815632
- Strict test caches: `data/processed/test1_cached_func_esm2_cdhit080_strict.pt` and `test2_cached_func_esm2_cdhit080_strict.pt`
- Conclusion: the full-data model remains stronger on identical strict subsets; CD-HIT retraining is not adopted.

## Figure Generation

```powershell
$env:MPLBACKEND="Agg"
& $py -B -m ghxtox.report_figures `
  --test1-predictions runs\plm_fusion_esm2_geometry_confidence\test1_predictions.csv `
  --test2-predictions runs\plm_fusion_esm2_geometry_confidence\test2_predictions.csv `
  --output-dir reports\figures `
  --threshold 0.85
```

Outputs:

- `reports/figures/default_model_evaluation.png`
- `reports/figures/default_model_evaluation.pdf`
- `reports/figures/roc_curve_points.csv`
- `reports/figures/pr_curve_points.csv`
- `reports/figures/figure_metadata.json`

## Verification

Final smoke-test command:

```powershell
$env:MPLBACKEND="Agg"
& $py -B -m pytest tests\test_smoke.py -q -p no:cacheprovider
```

Expected result at report generation: 27 tests passed. PyTorch may emit the known nested-tensor warning because Transformer layers use `norm_first=True`; it does not affect correctness.

