"""Validate the Reliable25 prompts and create matching rotated references.

Each supplied 300-bp prompt is located in its accession's GenBank sequence.
The full reference is rotated to put that exact prompt at position zero, so a
generated continuation can be compared with the corresponding natural tail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from Bio import SeqIO


ROOT = Path.cwd()
INPUTS = ROOT / "sae-arm" / "inputs"
PROMPTS = INPUTS / "prompts_reliable25_300bp.fa"
GENBANK = ROOT / "data" / "microviridae_records_20261111.gb"
REFERENCES = INPUTS / "reliable25_300bp_oriented_references.fna"
MANIFEST = INPUTS / "reliable25_300bp_prompt_manifest.json"
EXPECTED_COUNT = 25
PROMPT_LENGTH = 300


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks).upper()))
            name, chunks = line[1:].strip(), []
        else:
            if name is None:
                raise ValueError("FASTA sequence appears before the first header")
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    return records


def occurrences(sequence: str, prompt: str) -> list[int]:
    found: list[int] = []
    start = 0
    while True:
        position = sequence.find(prompt, start)
        if position < 0:
            return found
        found.append(position)
        start = position + 1


def main() -> None:
    if not PROMPTS.exists():
        raise SystemExit(f"Missing prompt FASTA: {PROMPTS}")
    if not GENBANK.exists():
        raise SystemExit(f"Missing source GenBank file: {GENBANK}")

    prompt_rows = fasta_records(PROMPTS)
    if len(prompt_rows) != EXPECTED_COUNT:
        raise SystemExit(f"Expected {EXPECTED_COUNT} prompts; found {len(prompt_rows)}")
    if len({name.split("__", 1)[0] for name, _ in prompt_rows}) != EXPECTED_COUNT:
        raise SystemExit("Prompt accessions must be unique")

    references = {record.id: record for record in SeqIO.parse(str(GENBANK), "genbank")}
    output_rows: list[tuple[str, str]] = []
    manifest_rows: list[dict[str, object]] = []

    for prompt_index, (header, prompt) in enumerate(prompt_rows):
        accession = header.split("__", 1)[0]
        if len(prompt) != PROMPT_LENGTH:
            raise SystemExit(f"{header}: expected {PROMPT_LENGTH} bp, got {len(prompt)}")
        if set(prompt) - set("ACGT"):
            raise SystemExit(f"{header}: prompt contains non-ACGT characters")
        if accession not in references:
            raise SystemExit(f"No GenBank record found for {accession}")

        record = references[accession]
        genome = str(record.seq).upper()
        positions = occurrences(genome, prompt)
        if len(positions) != 1:
            raise SystemExit(f"{accession}: expected one exact prompt match, found {positions}")
        prompt_start = positions[0]
        oriented = genome[prompt_start:] + genome[:prompt_start]
        if oriented[:PROMPT_LENGTH] != prompt:
            raise AssertionError(f"{accession}: rotation failed to preserve the prompt")

        anchor_products = []
        for feature in record.features:
            if feature.type != "CDS":
                continue
            start = int(feature.location.start)
            end = int(feature.location.end)
            if start <= prompt_start < end:
                anchor_products.append(str(feature.qualifiers.get("product", [""])[0]))

        output_rows.append((f"{accession} prompt_start={prompt_start}", oriented))
        manifest_rows.append({
            "prompt_index": prompt_index,
            "prompt_header": header,
            "accession": accession,
            "prompt_length_bp": len(prompt),
            "prompt_sha256": sha256(prompt),
            "genome_length_bp": len(genome),
            "generation_length_bp": len(genome) - len(prompt),
            "prompt_start_in_source_genome_0based": prompt_start,
            "prompt_matches_source_exactly": True,
            "oriented_reference_sha256": sha256(oriented),
            "source_cds_products_overlapping_prompt_start": anchor_products,
        })

    with REFERENCES.open("w", encoding="ascii", newline="\n") as handle:
        for header, sequence in output_rows:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")

    lengths = [int(row["genome_length_bp"]) for row in manifest_rows]
    generations = [int(row["generation_length_bp"]) for row in manifest_rows]
    manifest = {
        "status": "PASS",
        "prompt_file": PROMPTS.name,
        "prompt_count": len(prompt_rows),
        "prompt_length_bp": PROMPT_LENGTH,
        "reference_source": "data/microviridae_records_20261111.gb",
        "orientation_rule": "Rotate each accession's forward GenBank sequence so the exact prompt begins at position zero.",
        "reference_matches_verified": len(manifest_rows),
        "genome_length_range_bp": [min(lengths), max(lengths)],
        "generation_length_range_bp": [min(generations), max(generations)],
        "oriented_reference_fasta": REFERENCES.name,
        "prompts": manifest_rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "prompt_count": len(prompt_rows),
        "all_prompts_exactly_300bp": True,
        "exact_source_matches": len(manifest_rows),
        "genome_length_range_bp": manifest["genome_length_range_bp"],
        "generation_length_range_bp": manifest["generation_length_range_bp"],
        "outputs": [str(REFERENCES), str(MANIFEST)],
    }, indent=2))


if __name__ == "__main__":
    main()
