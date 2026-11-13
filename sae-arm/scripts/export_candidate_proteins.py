"""Export annotated proteins for screened, exact-deduplicated genomes."""

from __future__ import annotations

import csv
from pathlib import Path

from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CANDIDATES = DATA / "microviridae_screened_candidates_20261111.csv"
GENBANK = DATA / "microviridae_records_20261111.gb"
FASTA = DATA / "microviridae_screened_proteins_20261111.faa"
TABLE = DATA / "microviridae_screened_proteins_20261111.tsv"
GENOME_FASTA = DATA / "microviridae_screened_genomes_20261111.fna"


def main() -> None:
    with CANDIDATES.open(newline="", encoding="utf-8") as handle:
        candidates = {row["accession"] for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    with FASTA.open("w", encoding="ascii", newline="\n") as fasta, GENOME_FASTA.open(
        "w", encoding="ascii", newline="\n"
    ) as genomes:
        for record in SeqIO.parse(GENBANK, "genbank"):
            if record.id not in candidates:
                continue
            genome_sequence = str(record.seq).upper()
            genomes.write(f">{record.id}\n")
            for start in range(0, len(genome_sequence), 70):
                genomes.write(genome_sequence[start:start + 70] + "\n")
            cds_index = 0
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                translations = feature.qualifiers.get("translation")
                if not translations or not translations[0].strip():
                    continue
                cds_index += 1
                protein_id = f"{record.id}|cds{cds_index:02d}"
                sequence = "".join(translations[0].split()).rstrip("*").upper()
                if not sequence:
                    continue
                fasta.write(f">{protein_id}\n")
                for start in range(0, len(sequence), 70):
                    fasta.write(sequence[start:start + 70] + "\n")
                rows.append({
                    "protein_id": protein_id,
                    "accession": record.id,
                    "record_description": record.description,
                    "gene": feature.qualifiers.get("gene", [""])[0],
                    "product": feature.qualifiers.get("product", [""])[0],
                    "protein_accession": feature.qualifiers.get("protein_id", [""])[0],
                    "start_1based": int(feature.location.start) + 1,
                    "end_1based": int(feature.location.end),
                    "strand": feature.location.strand or 0,
                    "partial_start": feature.location.start.__class__.__name__ != "ExactPosition",
                    "partial_end": feature.location.end.__class__.__name__ != "ExactPosition",
                    "transl_table": feature.qualifiers.get("transl_table", ["1"])[0],
                    "protein_length_aa": len(sequence),
                    "protein_sequence": sequence,
                })
            candidates.remove(record.id)

    if candidates:
        raise SystemExit(f"Candidate accessions missing from GenBank: {len(candidates)}")

    fields = list(rows[0])
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} proteins from {len({r['accession'] for r in rows})} genomes")
    print(f"FASTA: {FASTA} ({FASTA.stat().st_size:,} bytes)")
    print(f"Table: {TABLE} ({TABLE.stat().st_size:,} bytes)")
    print(f"Genome FASTA: {GENOME_FASTA} ({GENOME_FASTA.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
