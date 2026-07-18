#!/usr/bin/env bash
set -euo pipefail

# Run from the GHXTox repository root after activating the server environment.
# Override MODEL_PATH with a downloaded local directory when Hugging Face access
# is unavailable, for example:
#   MODEL_PATH=/root/autodl-tmp/models/prot_t5_xl_half_uniref50-enc bash scripts/generate_prott5_pooled_autodl.sh

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

MODEL_PATH="${MODEL_PATH:-Rostlab/prot_t5_xl_half_uniref50-enc}"
OUTPUT="data/processed/train_prott5_official_mean.pt"
ARCHIVE="data/prott5_train_official_mean.tar.gz"

python -B -m ghxtox.plm_textcnn embed-prott5 \
  --fasta data/cdhit/input/train.fasta \
  --output "${OUTPUT}" \
  --model-path "${MODEL_PATH}" \
  --device cuda \
  --batch-size 1 \
  --save-every 100 \
  --pooling official_with_eos

tar -czf "${ARCHIVE}" "${OUTPUT}"
sha256sum "${OUTPUT}" "${ARCHIVE}"
ls -lh "${OUTPUT}" "${ARCHIVE}"
