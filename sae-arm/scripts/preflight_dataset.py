"""No-network/no-Modal validation for the prepared Microviridae inputs."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path

from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> None:
    with (DATA / "microviridae_screened_candidates_20261111.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        candidate_rows = list(csv.DictReader(handle))
    candidate_ids = {row["accession"] for row in candidate_rows}
    if len(candidate_ids) != len(candidate_rows):
        raise SystemExit("Candidate CSV contains duplicate accessions")

    genome_records = list(SeqIO.parse(DATA / "microviridae_screened_genomes_20261111.fna", "fasta"))
    genome_ids = {record.id for record in genome_records}
    if genome_ids != candidate_ids:
        raise SystemExit(
            f"Genome FASTA accession mismatch: missing={len(candidate_ids - genome_ids)}, "
            f"extra={len(genome_ids - candidate_ids)}"
        )
    invalid_genomes = [
        record.id for record in genome_records
        if not 4_000 <= len(record.seq) <= 6_500 or set(str(record.seq).upper()) - set("ACGT")
    ]
    if invalid_genomes:
        raise SystemExit(f"Invalid candidate genome sequences: {len(invalid_genomes)}")

    protein_records = list(SeqIO.parse(DATA / "microviridae_screened_proteins_20261111.faa", "fasta"))
    protein_ids = {record.id for record in protein_records}
    if len(protein_ids) != len(protein_records):
        raise SystemExit("Protein FASTA contains duplicate sequence IDs")
    protein_counts = Counter(record.id.split("|", 1)[0] for record in protein_records)
    if set(protein_counts) != candidate_ids:
        raise SystemExit("Protein FASTA accession set does not match candidate genomes")
    if min(protein_counts.values()) < 6:
        raise SystemExit("At least one candidate has fewer than six exported proteins")

    with (DATA / "microviridae_screened_proteins_20261111.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        table_rows = list(csv.DictReader(handle, delimiter="\t"))
    table_ids = {row["protein_id"] for row in table_rows}
    if table_ids != protein_ids or len(table_rows) != len(protein_records):
        raise SystemExit("Protein annotation TSV does not match protein FASTA")
    fasta_sequences = {record.id: str(record.seq) for record in protein_records}
    mismatched_proteins = sum(
        fasta_sequences[row["protein_id"]] != row["protein_sequence"]
        for row in table_rows
    )
    if mismatched_proteins:
        raise SystemExit(f"Protein sequence/table mismatches: {mismatched_proteins}")

    syntax_files = sorted((ROOT / "scripts").glob("*.py")) + sorted(
        (ROOT / "sae-arm" / "scripts").glob("*.py")
    )
    for path in syntax_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    summary = {
        "network_used": False,
        "modal_submitted_by_this_preflight": False,
        "candidate_genomes": len(candidate_ids),
        "protein_sequences": len(protein_records),
        "min_proteins_per_genome": min(protein_counts.values()),
        "max_proteins_per_genome": max(protein_counts.values()),
        "fasta_annotation_sequence_mismatches": mismatched_proteins,
        "python_scripts_syntax_checked": [path.name for path in syntax_files],
        "status": "PASS",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
