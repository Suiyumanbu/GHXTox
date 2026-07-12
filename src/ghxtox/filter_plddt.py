"""Filter processed peptide tensors by structure confidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _record_confidence(record: dict[str, Any], statistic: str) -> float:
    plddt = record.get("plddt")
    if not torch.is_tensor(plddt) or plddt.numel() == 0:
        return 0.0
    values = plddt.float().flatten()
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(values.median())
    if statistic == "min":
        return float(values.min())
    raise ValueError(f"Unsupported statistic={statistic!r}.")


def filter_processed(
    input_path: str | Path,
    output_path: str | Path,
    threshold: float,
    statistic: str = "mean",
) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    kept = [
        record
        for record in records
        if _record_confidence(record, statistic=statistic) >= threshold
    ]
    output = dict(payload)
    output["records"] = kept
    output["plddt_filter"] = {
        "statistic": statistic,
        "threshold": float(threshold),
        "input": str(input_path),
        "num_input": len(records),
        "num_output": len(kept),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return output["plddt_filter"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter a processed GHXTox .pt file by pLDDT confidence.")
    parser.add_argument("--input", required=True, help="Input processed .pt file.")
    parser.add_argument("--output", required=True, help="Output filtered .pt file.")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--statistic", choices=["mean", "median", "min"], default="mean")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = filter_processed(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        statistic=args.statistic,
    )
    print(
        f"Saved {summary['num_output']} / {summary['num_input']} records to {args.output} "
        f"({args.statistic} pLDDT >= {args.threshold:.3f})."
    )


if __name__ == "__main__":
    main()
