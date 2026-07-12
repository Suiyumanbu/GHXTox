# Bootstrap 95% Confidence Intervals

## Protocol

- Predictions: frozen default-model runs with seeds 42, 123, and 2025
- Decision threshold: fixed at 0.85
- Resampling: class-stratified percentile bootstrap
- Iterations: 5000
- Confidence level: 95%
- Bootstrap random seed: 2026
- Aggregate interval: hierarchical bootstrap that samples a model seed and then resamples positive and negative test examples separately
- Test1 composition: 1126 samples, 320 positive and 806 negative
- Test2 composition: 582 samples, 46 positive and 536 negative

## Aggregate Multi-Seed Intervals

| Dataset | MCC (95% CI) | F1 (95% CI) | AUROC (95% CI) | AUPRC (95% CI) |
| --- | --- | --- | --- | --- |
| Test1 | 0.8195 [0.7611, 0.8703] | 0.8678 [0.8239, 0.9063] | 0.9669 [0.9551, 0.9774] | 0.9336 [0.9078, 0.9557] |
| Test2 | 0.5782 [0.4419, 0.7109] | 0.6119 [0.4854, 0.7312] | 0.9225 [0.8766, 0.9599] | 0.6356 [0.4842, 0.7885] |

## Per-Seed Intervals

### Test1

| Seed | MCC (95% CI) | F1 (95% CI) | AUROC (95% CI) | AUPRC (95% CI) |
| ---: | --- | --- | --- | --- |
| 42 | 0.8458 [0.8110, 0.8787] | 0.8885 [0.8627, 0.9123] | 0.9699 [0.9592, 0.9793] | 0.9426 [0.9244, 0.9594] |
| 123 | 0.8222 [0.7856, 0.8583] | 0.8689 [0.8409, 0.8962] | 0.9649 [0.9533, 0.9751] | 0.9276 [0.9018, 0.9494] |
| 2025 | 0.7906 [0.7486, 0.8298] | 0.8459 [0.8138, 0.8756] | 0.9659 [0.9551, 0.9756] | 0.9306 [0.9097, 0.9504] |

### Test2

| Seed | MCC (95% CI) | F1 (95% CI) | AUROC (95% CI) | AUPRC (95% CI) |
| ---: | --- | --- | --- | --- |
| 42 | 0.6320 [0.5235, 0.7383] | 0.6602 [0.5607, 0.7551] | 0.9316 [0.8928, 0.9654] | 0.6969 [0.5784, 0.8114] |
| 123 | 0.5692 [0.4502, 0.6891] | 0.6042 [0.4946, 0.7115] | 0.9238 [0.8859, 0.9574] | 0.6217 [0.4899, 0.7609] |
| 2025 | 0.5334 [0.4099, 0.6526] | 0.5714 [0.4565, 0.6804] | 0.9120 [0.8652, 0.9526] | 0.5882 [0.4588, 0.7290] |

## Interpretation

Test1 ranking performance is stable, with relatively narrow AUROC and AUPRC intervals. Test2 intervals are substantially wider because only 46 positive examples are available and the model also varies across training seeds. Claims about exact test2 MCC or F1 should therefore use the aggregate interval rather than only the seed-42 point estimate.

The confidence intervals quantify uncertainty for the current datasets and training procedure. They do not correct for possible train-test sequence redundancy; sequence-similarity analysis remains the next validation step.

## Machine-Readable Results

- `runs/bootstrap_ci/test1_multiseed.json`
- `runs/bootstrap_ci/test2_multiseed.json`
