"""Resumable, residue-aligned ProtT5 feature extraction.

The cache is split into fixed FASTA-index shards.  Each shard stores a flat
``[sum(lengths), hidden_dim]`` tensor plus offsets, avoiding padding on disk
while preserving an exact mapping back to every peptide and residue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from ghxtox.plm_textcnn import _read_labeled_fasta
from ghxtox.utils import resolve_device


FORMAT_VERSION = 1


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _fasta_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_shard_payload(
    payload: dict[str, Any],
    expected_ids: list[str],
    expected_sequences: list[str],
    expected_labels: list[int],
    *,
    expected_dim: int | None = None,
) -> dict[str, int | str]:
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported or missing ProtT5 token-cache format_version.")
    if payload.get("sample_ids") != expected_ids:
        raise ValueError("Shard sample IDs do not match the FASTA slice.")
    if payload.get("sequences") != expected_sequences:
        raise ValueError("Shard sequences do not match the FASTA slice.")
    labels = payload.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.long().tolist() != expected_labels:
        raise ValueError("Shard labels do not match the FASTA slice.")
    embeddings = payload.get("embeddings")
    offsets = payload.get("offsets")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("Shard embeddings must be a rank-2 tensor.")
    if embeddings.dtype not in {torch.float16, torch.float32}:
        raise ValueError("Shard embeddings must use float16 or float32 storage.")
    if not isinstance(offsets, torch.Tensor) or offsets.ndim != 1:
        raise ValueError("Shard offsets must be a rank-1 tensor.")
    expected_offsets = [0]
    for sequence in expected_sequences:
        expected_offsets.append(expected_offsets[-1] + len(sequence))
    if offsets.long().tolist() != expected_offsets:
        raise ValueError("Shard offsets do not match peptide residue lengths.")
    if embeddings.shape[0] != expected_offsets[-1]:
        raise ValueError("Shard embedding rows do not equal the residue count.")
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected feature dimension {embeddings.shape[1]}; expected {expected_dim}."
        )
    if not torch.isfinite(embeddings.float()).all():
        raise ValueError("Shard embeddings contain NaN or infinity.")
    return {
        "num_samples": len(expected_ids),
        "num_residues": expected_offsets[-1],
        "feature_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype).replace("torch.", ""),
    }


def extract_prott5_tokens(
    fasta_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str,
    device_name: str = "cuda",
    batch_size: int = 1,
    shard_size: int = 256,
    storage_dtype: str = "float16",
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Extract true residue tokens (EOS and padding excluded) into resumable shards."""

    if batch_size < 1 or shard_size < 1:
        raise ValueError("batch_size and shard_size must be positive.")
    storage_dtypes = {"float16": torch.float16, "float32": torch.float32}
    if storage_dtype not in storage_dtypes:
        raise ValueError("storage_dtype must be 'float16' or 'float32'.")

    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ProtT5 extraction requires transformers and sentencepiece."
        ) from exc

    sample_ids, sequences, labels = _read_labeled_fasta(fasta_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(len(sequences) / shard_size)
    fasta_hash = _fasta_sha256(fasta_path)

    pending: list[int] = []
    feature_dim: int | None = None
    for shard_index in range(num_shards):
        start = shard_index * shard_size
        stop = min(start + shard_size, len(sequences))
        shard_path = output_path / f"shard_{shard_index:05d}.pt"
        if not shard_path.exists():
            pending.append(shard_index)
            continue
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        report = _validate_shard_payload(
            payload,
            sample_ids[start:stop],
            sequences[start:stop],
            labels[start:stop],
            expected_dim=feature_dim,
        )
        feature_dim = int(report["feature_dim"])
        metadata = payload.get("metadata", {})
        if metadata.get("fasta_sha256") != fasta_hash or metadata.get("model_path") != model_path:
            raise ValueError(
                f"Existing {shard_path.name} was generated from another FASTA or model."
            )

    if pending:
        device = resolve_device(device_name)
        model_dtype = torch.float16 if device.type == "cuda" else torch.float32
        tokenizer = T5Tokenizer.from_pretrained(
            model_path, do_lower_case=False, local_files_only=local_files_only
        )
        model = T5EncoderModel.from_pretrained(
            model_path, torch_dtype=model_dtype, local_files_only=local_files_only
        )
        model.eval().to(device)
        target_dtype = storage_dtypes[storage_dtype]

        with torch.inference_mode():
            for shard_index in pending:
                start = shard_index * shard_size
                stop = min(start + shard_size, len(sequences))
                shard_embeddings: list[torch.Tensor] = []
                for batch_start in range(start, stop, batch_size):
                    batch_stop = min(batch_start + batch_size, stop)
                    raw_sequences = sequences[batch_start:batch_stop]
                    encoded = tokenizer(
                        [" ".join(sequence) for sequence in raw_sequences],
                        add_special_tokens=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    hidden = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    ).last_hidden_state
                    for local_index, sequence in enumerate(raw_sequences):
                        token_count = int(encoded["attention_mask"][local_index].sum().item())
                        if token_count < len(sequence) + 1:
                            raise ValueError(
                                "Tokenizer did not produce one token per residue plus EOS for "
                                f"{sample_ids[batch_start + local_index]!r}."
                            )
                        residue_tensor = hidden[local_index, : len(sequence)].detach().to(
                            device="cpu", dtype=target_dtype
                        )
                        shard_embeddings.append(residue_tensor)

                shard_sequences = sequences[start:stop]
                offsets = [0]
                for sequence in shard_sequences:
                    offsets.append(offsets[-1] + len(sequence))
                embedding_tensor = torch.cat(shard_embeddings, dim=0)
                feature_dim = int(embedding_tensor.shape[1])
                payload = {
                    "format_version": FORMAT_VERSION,
                    "embeddings": embedding_tensor,
                    "offsets": torch.tensor(offsets, dtype=torch.long),
                    "labels": torch.tensor(labels[start:stop], dtype=torch.long),
                    "sample_ids": sample_ids[start:stop],
                    "sequences": shard_sequences,
                    "metadata": {
                        "model_path": model_path,
                        "source_fasta": str(fasta_path),
                        "fasta_sha256": fasta_hash,
                        "first_source_index": start,
                        "last_source_index_exclusive": stop,
                        "pooling": "none; one vector per true residue",
                        "eos_included": False,
                        "padding_included": False,
                        "storage_dtype": storage_dtype,
                    },
                }
                _atomic_torch_save(payload, output_path / f"shard_{shard_index:05d}.pt")
                print(
                    f"shard={shard_index + 1}/{num_shards} "
                    f"samples={stop}/{len(sequences)} residues={offsets[-1]}"
                )

    summary = validate_token_cache(fasta_path, output_path, model_path=model_path)
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_fasta": str(fasta_path),
        "fasta_sha256": fasta_hash,
        "model_path": model_path,
        "num_samples": len(sequences),
        "num_residues": sum(map(len, sequences)),
        "num_shards": num_shards,
        "shard_size": shard_size,
        "feature_dim": summary["feature_dim"],
        "storage_dtype": summary["storage_dtype"],
        "eos_included": False,
        "padding_included": False,
        "shards": [f"shard_{index:05d}.pt" for index in range(num_shards)],
    }
    _atomic_json_save(manifest, output_path / "manifest.json")
    return manifest


def validate_token_cache(
    fasta_path: str | Path,
    input_dir: str | Path,
    *,
    model_path: str | None = None,
) -> dict[str, Any]:
    """Fully validate token order, labels, offsets, dimensions and finite values."""

    sample_ids, sequences, labels = _read_labeled_fasta(fasta_path)
    input_path = Path(input_dir)
    shard_paths = sorted(input_path.glob("shard_*.pt"))
    if not shard_paths:
        raise ValueError(f"No shard_*.pt files found under {input_path}.")
    cursor = 0
    total_residues = 0
    feature_dim: int | None = None
    dtype: str | None = None
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        count = len(payload.get("sample_ids", []))
        stop = cursor + count
        if stop > len(sample_ids):
            raise ValueError(f"{shard_path.name} extends beyond the FASTA length.")
        report = _validate_shard_payload(
            payload,
            sample_ids[cursor:stop],
            sequences[cursor:stop],
            labels[cursor:stop],
            expected_dim=feature_dim,
        )
        if model_path is not None and payload.get("metadata", {}).get("model_path") != model_path:
            raise ValueError(f"{shard_path.name} was generated by another model.")
        feature_dim = int(report["feature_dim"])
        current_dtype = str(report["dtype"])
        if dtype is not None and current_dtype != dtype:
            raise ValueError("All shards must use the same storage dtype.")
        dtype = current_dtype
        total_residues += int(report["num_residues"])
        cursor = stop
    if cursor != len(sample_ids):
        raise ValueError(f"Cache contains {cursor} samples; FASTA contains {len(sample_ids)}.")
    summary = {
        "status": "ok",
        "input_dir": str(input_path),
        "num_samples": cursor,
        "num_residues": total_residues,
        "num_shards": len(shard_paths),
        "feature_dim": feature_dim,
        "storage_dtype": dtype,
        "eos_included": False,
        "padding_included": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="Generate resumable token-level shards.")
    extract.add_argument("--fasta", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--model-path", required=True)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--shard-size", type=int, default=256)
    extract.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    extract.add_argument("--local-files-only", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate a complete token cache.")
    validate.add_argument("--fasta", required=True)
    validate.add_argument("--input-dir", required=True)
    validate.add_argument("--model-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        result = extract_prott5_tokens(
            args.fasta,
            args.output_dir,
            model_path=args.model_path,
            device_name=args.device,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            storage_dtype=args.storage_dtype,
            local_files_only=args.local_files_only,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        validate_token_cache(args.fasta, args.input_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
