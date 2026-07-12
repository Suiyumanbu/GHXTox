"""Attach pretrained protein language model residue embeddings to GHXTox records."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from ghxtox.utils import DEFAULT_DEVICE, DEFAULT_TRAIN_PROCESSED, resolve_device


ESM2_MODELS: dict[str, tuple[str, int, int]] = {
    "esm2_t6_8M_UR50D": ("esm2_t6_8M_UR50D", 6, 320),
    "esm2_t12_35M_UR50D": ("esm2_t12_35M_UR50D", 12, 480),
    "esm2_t30_150M_UR50D": ("esm2_t30_150M_UR50D", 30, 640),
    "esm2_t33_650M_UR50D": ("esm2_t33_650M_UR50D", 33, 1280),
    "esm2_t36_3B_UR50D": ("esm2_t36_3B_UR50D", 36, 2560),
}
PROTT5_MODELS: dict[str, tuple[str, int]] = {
    "prot_t5_xl_half_uniref50-enc": ("models/prot_t5_xl_half_uniref50-enc", 1024),
}


def _load_esm2(model_name: str, device: torch.device) -> tuple[Any, Any, int, int]:
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "ESM2 embedding generation requires the optional 'fair-esm' package. "
            "Install it with: pip install fair-esm"
        ) from exc

    if model_name not in ESM2_MODELS:
        choices = ", ".join(sorted(ESM2_MODELS))
        raise ValueError(f"Unsupported ESM2 model '{model_name}'. Available: {choices}")
    loader_name, layer, dim = ESM2_MODELS[model_name]
    loader = getattr(esm.pretrained, loader_name)
    model, alphabet = loader()
    model.eval().to(device)
    return model, alphabet, layer, dim


def _load_prott5(model_name: str, device: torch.device, model_path: str | Path | None = None) -> tuple[Any, Any, int]:
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ProtT5 embedding generation requires transformers, sentencepiece, and protobuf."
        ) from exc

    if model_name not in PROTT5_MODELS:
        choices = ", ".join(sorted(PROTT5_MODELS))
        raise ValueError(f"Unsupported ProtT5 model '{model_name}'. Available: {choices}")
    default_path, dim = PROTT5_MODELS[model_name]
    source = str(model_path or default_path)
    tokenizer = T5Tokenizer.from_pretrained(source)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = T5EncoderModel.from_pretrained(source, torch_dtype=dtype)
    model.eval().to(device)
    return model, tokenizer, dim


def _format_prott5_sequence(sequence: str) -> str:
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return " ".join(sequence)


def _cache_name(sample_id: str, sequence: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(f"{sample_id}\0{sequence}".encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}.pt"


def _load_cached_embedding(cache_dir: Path, sample_id: str, sequence: str, model_name: str) -> torch.Tensor | None:
    path = cache_dir / _cache_name(sample_id, sequence)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        return payload.float()
    if payload.get("model") != model_name:
        return None
    return payload["embedding"].float()


def _save_cached_embedding(cache_dir: Path, sample_id: str, sequence: str, embedding: torch.Tensor, model_name: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _cache_name(sample_id, sequence)
    torch.save(
        {
            "sample_id": sample_id,
            "sequence": sequence,
            "model": model_name,
            "embedding": embedding.cpu().float(),
        },
        path,
    )


def _validate_embedding(record: dict[str, Any], embedding: torch.Tensor, expected_dim: int) -> torch.Tensor:
    sequence = record["sequence"]
    if embedding.ndim != 2 or embedding.shape[-1] != expected_dim:
        raise ValueError(
            f"Invalid PLM embedding for {record['sample_id']}: expected (*, {expected_dim}), got {tuple(embedding.shape)}"
        )
    if embedding.shape[0] < len(sequence):
        raise ValueError(
            f"PLM embedding is shorter than sequence for {record['sample_id']}: "
            f"embedding={embedding.shape[0]}, sequence={len(sequence)}"
        )
    return embedding[: len(sequence)].contiguous().float()


def attach_esm2_embeddings(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    device_name: str,
    batch_size: int,
    cache_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    cache_path = Path(cache_dir) if cache_dir else None
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    records: list[dict[str, Any]] = payload["records"]
    device = resolve_device(device_name)
    model, alphabet, layer, expected_dim = _load_esm2(model_name, device)
    batch_converter = alphabet.get_batch_converter()

    attached = 0
    loaded_from_cache = 0
    generated = 0
    pending: list[dict[str, Any]] = []

    for record in records:
        if skip_existing and torch.is_tensor(record.get("plm_features")):
            record["plm_features"] = _validate_embedding(record, record["plm_features"], expected_dim)
            record["plm_model"] = model_name
            attached += 1
            continue
        cached = _load_cached_embedding(cache_path, record["sample_id"], record["sequence"], model_name) if cache_path else None
        if cached is not None:
            record["plm_features"] = _validate_embedding(record, cached, expected_dim)
            record["plm_model"] = model_name
            attached += 1
            loaded_from_cache += 1
        else:
            pending.append(record)

    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            labels_and_sequences = [(item["sample_id"], item["sequence"]) for item in chunk]
            _, _, tokens = batch_converter(labels_and_sequences)
            tokens = tokens.to(device)
            output = model(tokens, repr_layers=[layer], return_contacts=False)
            representations = output["representations"][layer].detach().cpu()
            for item, embedding in zip(chunk, representations):
                seq_len = len(item["sequence"])
                residue_embedding = embedding[1 : seq_len + 1].contiguous().float()
                item["plm_features"] = _validate_embedding(item, residue_embedding, expected_dim)
                item["plm_model"] = model_name
                if cache_path:
                    _save_cached_embedding(cache_path, item["sample_id"], item["sequence"], item["plm_features"], model_name)
                attached += 1
                generated += 1
            print(f"attached={attached}/{len(records)} generated={generated} cached={loaded_from_cache}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload["records"] = records
    payload["plm_model"] = model_name
    payload["plm_embedding_dim"] = expected_dim
    torch.save(payload, output_path)
    return {"attached": attached, "generated": generated, "cached": loaded_from_cache, "dim": expected_dim}


def attach_prott5_embeddings(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    device_name: str,
    batch_size: int,
    cache_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    cache_path = Path(cache_dir) if cache_dir else None
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    records: list[dict[str, Any]] = payload["records"]
    device = resolve_device(device_name)
    model, tokenizer, expected_dim = _load_prott5(model_name, device, model_path=model_path)

    attached = 0
    loaded_from_cache = 0
    generated = 0
    pending: list[dict[str, Any]] = []

    for record in records:
        if skip_existing and torch.is_tensor(record.get("plm_features")):
            record["plm_features"] = _validate_embedding(record, record["plm_features"], expected_dim)
            record["plm_model"] = model_name
            attached += 1
            continue
        cached = _load_cached_embedding(cache_path, record["sample_id"], record["sequence"], model_name) if cache_path else None
        if cached is not None:
            record["plm_features"] = _validate_embedding(record, cached, expected_dim)
            record["plm_model"] = model_name
            attached += 1
            loaded_from_cache += 1
        else:
            pending.append(record)

    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            sequences = [_format_prott5_sequence(item["sequence"]) for item in chunk]
            encoded = tokenizer(
                sequences,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            representations = output.last_hidden_state.detach().cpu()
            for item, embedding in zip(chunk, representations):
                seq_len = len(item["sequence"])
                residue_embedding = embedding[:seq_len].contiguous().float()
                item["plm_features"] = _validate_embedding(item, residue_embedding, expected_dim)
                item["plm_model"] = model_name
                if cache_path:
                    _save_cached_embedding(cache_path, item["sample_id"], item["sequence"], item["plm_features"], model_name)
                attached += 1
                generated += 1
            print(f"attached={attached}/{len(records)} generated={generated} cached={loaded_from_cache}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload["records"] = records
    payload["plm_model"] = model_name
    payload["plm_embedding_dim"] = expected_dim
    torch.save(payload, output_path)
    return {"attached": attached, "generated": generated, "cached": loaded_from_cache, "dim": expected_dim}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach pretrained PLM residue embeddings to a GHXTox processed .pt file.")
    parser.add_argument("--input", default=DEFAULT_TRAIN_PROCESSED, help="Input processed .pt file.")
    parser.add_argument("--output", default="data/processed/train_cached_func_esm2.pt", help="Output processed .pt file.")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D", choices=sorted({**ESM2_MODELS, **PROTT5_MODELS}))
    parser.add_argument("--model-path", default=None, help="Local model directory for HuggingFace PLMs such as ProtT5.")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", default=None, help="Per-sample embedding cache directory. Defaults to data/plm/<model>.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Regenerate embeddings already stored in records.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    common_kwargs = {
        "input_path": args.input,
        "output_path": args.output,
        "model_name": args.model,
        "device_name": args.device,
        "batch_size": args.batch_size,
        "cache_dir": args.cache_dir or f"data/plm/{args.model}",
        "skip_existing": not args.no_skip_existing,
    }
    if args.model in ESM2_MODELS:
        stats = attach_esm2_embeddings(**common_kwargs)
    elif args.model in PROTT5_MODELS:
        stats = attach_prott5_embeddings(**common_kwargs, model_path=args.model_path)
    else:
        raise ValueError(f"Unsupported PLM model: {args.model}")
    print(
        f"PLM embeddings saved to {args.output}: attached={stats['attached']}, "
        f"generated={stats['generated']}, cached={stats['cached']}, dim={stats['dim']}"
    )


if __name__ == "__main__":
    main()
