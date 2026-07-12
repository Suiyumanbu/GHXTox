# Train-Test Sequence Similarity Audit

## Status

The local machine does not have `cd-hit` or `cd-hit-2d` installed. The completed results below are therefore a conservative Biopython global-alignment pre-audit, not native CD-HIT results. Unique-ID CD-HIT inputs, manifests, and a Linux execution script have been prepared for the native follow-up.

## Pre-Audit Protocol

- Training set: 6387 peptides, 1818 positive and 4569 negative
- Test1: 1126 peptides, 320 positive and 806 negative
- Test2: 582 peptides, 46 positive and 536 negative
- Sequence lengths: 11 to 50 residues in every split
- Alignment: Biopython global pairwise alignment
- Identity: identical aligned residues divided by the shorter sequence length
- Candidate prefilter: shared 2-mer
- Thresholds: 0.90 and 0.80
- Strict subset rule: retain a test peptide only when its maximum identity to every training peptide is below the threshold

The shared 2-mer filter is used instead of a 3-mer filter because a 3-mer filter can miss short peptides at exactly 0.80 identity.

## Similarity Counts

| Dataset | Threshold | High-similarity | Same label | Label conflict | Retained | Retained positive | Retained negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test1 | >=0.90 | 121 | 107 | 14 | 1005 | 286 | 719 |
| Test1 | >=0.80 | 698 | 634 | 64 | 428 | 132 | 296 |
| Test2 | >=0.90 | 76 | 66 | 10 | 506 | 35 | 471 |
| Test2 | >=0.80 | 374 | 334 | 40 | 208 | 15 | 193 |

The initial audit counted 8 test1 and 5 test2 cases with identity 1.0 under the shorter-sequence denominator. These are full shorter-sequence matches and may contain terminal or internal insertions; they are not necessarily exact string duplicates. Exact string matches are reported separately in the machine-readable summaries.

## Strict-Subset Performance

All values are the mean and sample standard deviation over model seeds 42, 123, and 2025. The model and decision threshold 0.85 are frozen; no retraining or test-dependent tuning is performed.

### Test1

| Subset | Samples | Positive | BACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 1126 | 320 | 0.8999 +/- 0.0174 | 0.8678 +/- 0.0213 | 0.8195 +/- 0.0276 | 0.9669 +/- 0.0027 | 0.9336 +/- 0.0079 |
| Identity <0.90 | 1005 | 286 | 0.9005 +/- 0.0198 | 0.8676 +/- 0.0246 | 0.8188 +/- 0.0321 | 0.9672 +/- 0.0036 | 0.9335 +/- 0.0099 |
| Identity <0.80 | 428 | 132 | 0.8659 +/- 0.0208 | 0.8225 +/- 0.0284 | 0.7494 +/- 0.0397 | 0.9333 +/- 0.0060 | 0.8908 +/- 0.0180 |

### Test2

| Subset | Samples | Positive | BACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 582 | 46 | 0.8086 +/- 0.0349 | 0.6119 +/- 0.0449 | 0.5782 +/- 0.0499 | 0.9225 +/- 0.0099 | 0.6356 +/- 0.0557 |
| Identity <0.90 | 506 | 35 | 0.7872 +/- 0.0440 | 0.5566 +/- 0.0587 | 0.5236 +/- 0.0650 | 0.9248 +/- 0.0120 | 0.5945 +/- 0.0633 |
| Identity <0.80 | 208 | 15 | 0.7552 +/- 0.0365 | 0.4719 +/- 0.0402 | 0.4321 +/- 0.0469 | 0.8639 +/- 0.0415 | 0.4704 +/- 0.0537 |

## Interpretation

Removing test peptides with close training neighbors has little effect on test1 at the 0.90 threshold, but materially reduces test2 MCC, F1, and AUPRC. The drop is larger at 0.80 in both test sets. The current full-split results therefore benefit from sequence redundancy and should be accompanied by strict-subset results when compared with papers using CD-HIT-controlled protocols.

The identity<0.80 test2 subset contains only 15 positives, so its decision metrics have high sampling uncertainty. It is useful as a stress test, not as a replacement for the original benchmark.

## Native CD-HIT Follow-Up

Prepared files:

- `data/cdhit/input/train.fasta`
- `data/cdhit/input/test1.fasta`
- `data/cdhit/input/test2.fasta`
- Corresponding `*_manifest.csv` files
- `scripts/run_cdhit_redundancy.sh`

Run on a Linux server with CD-HIT installed:

```bash
bash scripts/run_cdhit_redundancy.sh data/cdhit/input runs/cdhit
```

The first server run used `cd-hit-2d`, but inspection showed directional length misses: a longer test peptide could remain even when a shorter training peptide matched all of its residues. Those test subsets must not be used for model evaluation. The corrected script clusters each combined train+test FASTA, after which strict test subsets are extracted only from clusters containing no training member. Native `.clstr` files and retained FASTA counts must be reported separately from this pre-audit.

The training-set clustering outputs from the first native run remain valid. At 0.90, 6380 representatives remain (1816 positive and 4564 negative); at 0.80, 5183 remain (1462 positive and 3721 negative). The 0.90 and 0.80 training cluster files contain 7 and 11 mixed-label clusters, respectively. Representative selection therefore removes a small number of examples whose close sequence neighbors carry conflicting labels.

## Native Combined-Clustering Results

The corrected train+test combined clustering has been completed on CD-HIT 4.8.1. A strict test peptide is retained only when its cluster contains no training peptide.

| Threshold | Training representatives | Test1 strict | Test1 positive | Test2 strict | Test2 positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 6380 | 1123 | 318 | 580 | 45 |
| 0.80 | 5183 | 771 | 200 | 425 | 37 |

The originally generated `cd-hit-2d` low-similarity FASTA files are invalid for final evaluation because of directional length misses. Only the combined-clustering `test*_strict.fasta` files are used below.

## CD-HIT 0.80 Retraining

The default architecture was retrained on the 5183 CD-HIT 0.80 training representatives using seed 42, validation-MCC checkpoint selection, and the same fixed test threshold 0.85. The best checkpoint was epoch 14 with validation MCC 0.8156.

### Full Test Sets

| Model | Test1 MCC | Test1 F1 | Test1 AUPRC | Test2 MCC | Test2 F1 | Test2 AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-data default | 0.8458 | 0.8885 | 0.9426 | 0.6320 | 0.6602 | 0.6969 |
| CD-HIT 0.80 retrained | 0.8037 | 0.8528 | 0.9242 | 0.5426 | 0.5778 | 0.5936 |

### Same CD-HIT 0.80 Strict Test Sets

| Model | Test1 MCC | Test1 F1 | Test1 AUPRC | Test2 MCC | Test2 F1 | Test2 AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-data default | 0.7925 | 0.8456 | 0.9064 | 0.5941 | 0.6279 | 0.6606 |
| CD-HIT 0.80 retrained | 0.7562 | 0.8118 | 0.8852 | 0.4843 | 0.5278 | 0.5421 |

The full-data model remains stronger even when both checkpoints are evaluated on the identical strict subsets. Removing training redundancy therefore does not improve extrapolation in this experiment; the reduction from 6387 to 5183 training peptides instead lowers recall and ranking performance. The CD-HIT 0.80 model is retained as a protocol-alignment experiment and does not replace the default model.

Training a separate CD-HIT 0.90 model is not warranted: only seven training representatives are removed, while the strict test sets differ by only three test1 and two test2 examples. The existing default result is effectively the 0.90 reference at this data scale.

## Machine-Readable Results

- `runs/sequence_similarity/test1_summary.json`
- `runs/sequence_similarity/test2_summary.json`
- `runs/sequence_similarity/test1_rows.csv`
- `runs/sequence_similarity/test2_rows.csv`
- `runs/sequence_similarity/test1_subset_metrics.json`
- `runs/sequence_similarity/test2_subset_metrics.json`
