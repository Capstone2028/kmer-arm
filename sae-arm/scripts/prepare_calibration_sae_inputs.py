"""Bundle calibration-only genomes and their five annotated family regions."""

from __future__ import annotations

import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "sae-arm" / "results"
SPLIT = RESULTS / "microviridae_split_20261112.tsv"
GENOME_FASTAS = {
    "calibration": RESULTS / "microviridae_calibration_genomes_20261112.fna",
    "test": RESULTS / "microviridae_test_genomes_20261112.fna",
}
PROTEINS = DATA / "microviridae_screened_proteins_20261111.tsv"
CLUSTERS = RESULTS / "microviridae_protein_families_30pct_20261111.tsv"

FAMILY_IDS = (
    "MT310288.1|cds01",
    "MW697641.1|cds06",
    "MZ089757.1|cds01",
    "MT310270.1|cds06",
    "MK249233.1|cds03",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main(split_name: str) -> None:
    expected_genomes = {"calibration": 30, "test": 60}[split_name]
    genome_fasta = GENOME_FASTAS[split_name]
    output = RESULTS / f"microviridae_{split_name}_sae_inputs_20261112.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    split_rows = [row for row in read_tsv(SPLIT) if row["split"] == split_name]
    if len(split_rows) != expected_genomes:
        raise SystemExit(f"Expected {expected_genomes} {split_name} genomes; found {len(split_rows)}")

    calibration_ids = {row["accession"] for row in split_rows}
    family_members: dict[str, str] = {}
    for representative, member in csv.reader(CLUSTERS.open(encoding="utf-8"), delimiter="\t"):
        if representative in FAMILY_IDS:
            prior = family_members.setdefault(member, representative)
            if prior != representative:
                raise SystemExit(f"Protein belongs to two target families: {member}")

    gene_regions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_tsv(PROTEINS):
        accession = row["accession"]
        protein_id = row["protein_id"]
        family_id = family_members.get(protein_id)
        if accession not in calibration_ids or family_id is None:
            continue
        start_1based = int(row["start_1based"])
        end_1based = int(row["end_1based"])
        gene_regions[accession].append({
            "family_id": family_id,
            "protein_id": protein_id,
            "product": row["product"],
            "start_0": start_1based - 1,
            "end_0": end_1based,
            "strand": int(row["strand"]),
            "partial_start": row["partial_start"].lower() == "true",
            "partial_end": row["partial_end"].lower() == "true",
            "translation_table": int(row["transl_table"]),
        })

    records = {record.id: str(record.seq).upper() for record in SeqIO.parse(genome_fasta, "fasta")}
    if set(records) != calibration_ids:
        raise SystemExit("Calibration FASTA IDs do not match the calibration split")

    bundled = []
    for accession in sorted(calibration_ids):
        regions = sorted(gene_regions[accession], key=lambda item: str(item["family_id"]))
        represented = [str(region["family_id"]) for region in regions]
        if len(regions) != len(FAMILY_IDS) or set(represented) != set(FAMILY_IDS):
            raise SystemExit(f"{accession}: expected one gene from each of five families; got {represented}")
        sequence = records[accession]
        for region in regions:
            if not (0 <= int(region["start_0"]) < int(region["end_0"]) <= len(sequence)):
                raise SystemExit(f"{accession}: invalid coordinates for {region['protein_id']}")
        bundled.append({
            "accession": accession,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "family_regions": regions,
        })

    payload = {
        "created_on": "2026-11-12",
        "dataset_note": "Fresh NCBI snapshot; new calibration split, not the unavailable earlier raw split.",
        "split": split_name,
        "genome_count": len(bundled),
        "family_ids": list(FAMILY_IDS),
        "family_gene_count": sum(len(record["family_regions"]) for record in bundled),
        "coordinate_convention": "0-based, half-open coordinates on the submitted forward genome sequence.",
        "records": bundled,
    }
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "genome_count": len(bundled),
        "family_gene_count": payload["family_gene_count"],
        "output": str(output),
        "length_range_bp": [
            min(record["sequence_length"] for record in bundled),
            max(record["sequence_length"] for record in bundled),
        ],
        "partial_family_gene_count": sum(
            bool(region["partial_start"] or region["partial_end"])
            for record in bundled for region in record["family_regions"]
        ),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=GENOME_FASTAS, default="calibration")
    main(parser.parse_args().split)
