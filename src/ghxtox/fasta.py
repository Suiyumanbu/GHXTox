"""FASTA parsing utilities with label extraction from headers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class FastaRecord:
    sample_id: str
    sequence: str
    label: int | None
    header: str


def _parse_label(header: str) -> int | None:
    """Extract a binary label from common header conventions.

    Supported examples:
    >peptide|1
    >sequence_12|0
    >id label=1
    >id toxicity: negative
    """

    text = header.strip()
    pipe_tail = text.rsplit("|", 1)[-1].strip()
    if pipe_tail in {"0", "1"}:
        return int(pipe_tail)

    match = re.search(r"(?:label|class|toxicity)\s*[:=]\s*([01])\b", text, re.I)
    if match:
        return int(match.group(1))

    lowered = text.lower()
    if re.search(r"\b(non[-_ ]?toxic|negative|neg)\b", lowered):
        return 0
    if re.search(r"\b(toxic|positive|pos)\b", lowered):
        return 1
    return None


def _parse_id(header: str, index: int) -> str:
    first = header.strip().split()[0]
    if "|" in first:
        first = first.split("|", 1)[0]
    first = first.lstrip(">")
    return first or f"sample_{index}"


def read_fasta(path: str | Path) -> list[FastaRecord]:
    path = Path(path)
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if header is None:
            return
        sequence = "".join(chunks).strip().upper().replace(" ", "")
        if sequence:
            records.append(
                FastaRecord(
                    sample_id=_parse_id(header, len(records) + 1),
                    sequence=sequence,
                    label=_parse_label(header),
                    header=header,
                )
            )
        header = None
        chunks = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                chunks.append(line)
    flush()
    return records
