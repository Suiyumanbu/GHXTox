#!/usr/bin/env bash
set -euo pipefail

# Run from the GHXTox repository root. A local model directory is strongly
# recommended so all three splits use exactly the same frozen encoder.
# Example:
#   MODEL_PATH=/root/autodl-tmp/models/prot_t5_xl_half_uniref50-enc \
#     bash scripts/generate_prott5_tokens_autodl.sh

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/prot_t5_xl_half_uniref50-enc}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHARD_SIZE="${SHARD_SIZE:-256}"
OUTPUT_ROOT="data/processed/prott5_tokens"
ARCHIVE="data/prott5_tokens_fp16.tar.gz"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: ProtT5 model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

for split in train test1 test2; do
  python -B -m ghxtox.prott5_tokens extract \
    --fasta "data/cdhit/input/${split}.fasta" \
    --output-dir "${OUTPUT_ROOT}/${split}" \
    --model-path "${MODEL_PATH}" \
    --device cuda \
    --batch-size "${BATCH_SIZE}" \
    --shard-size "${SHARD_SIZE}" \
    --storage-dtype float16 \
    --local-files-only
done

tar -czf "${ARCHIVE}" "${OUTPUT_ROOT}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
du -sh "${OUTPUT_ROOT}"
ls -lh "${ARCHIVE}" "${ARCHIVE}.sha256"
