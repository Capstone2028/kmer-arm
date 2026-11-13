"""Inventory NCBI Microviridae GenBank records and apply basic sequence filters."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from Bio import SeqIO


ACCESSIONS = Path("data/microviridae_accessions_20261111.txt")
GENBANK = Path("data/microviridae_records_20261111.gb")
INVENTORY = Path("data/microviridae_candidate_inventory_20261111.csv")
SUMMARY = Path("data/microviridae_candidate_summary_20261111.json")
CANDIDATES = Path("data/microviridae_screened_candidates_20261111.csv")


def main() -> None:
    expected = {line.strip() for line in ACCESSIONS.read_text(encoding="ascii").splitlines() if line.strip()}
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    lengths: list[int] = []
    seen: set[str] = set()
    hashes: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    product_genomes: Counter[str] = Counter()
    candidate_product_genomes: Counter[str] = Counter()
    candidate_by_hash: dict[str, dict[str, object]] = {}

    for record in SeqIO.parse(GENBANK, "genbank"):
        sequence = str(record.seq).upper()
        hashes[hashlib.sha256(sequence.encode("ascii")).hexdigest()] += 1
        cds = [feature for feature in record.features if feature.type == "CDS"]
        cds_lengths = [int(len(feature.location)) for feature in cds]
        valid_bases = set(sequence) <= set("ACGT")
        valid_length = 4_000 <= len(sequence) <= 6_500
        counts["records"] += 1
        counts["valid_length_and_acgt"] += int(valid_bases and valid_length)
        counts["all_acgt"] += int(valid_bases)
        counts["length_range"] += int(valid_length)
        counts["six_cds_ge_150nt"] += int(sum(n >= 150 for n in cds_lengths) >= 6)
        counts["six_translated_cds_ge_150nt"] += int(
            sum(n >= 150 and "translation" in feature.qualifiers
                for n, feature in zip(cds_lengths, cds)) >= 6
        )
        counts["circular"] += int(record.annotations.get("topology") == "circular")
        counts["linear"] += int(record.annotations.get("topology") == "linear")
        lengths.append(len(sequence))
        seen.add(record.id)

        gene_labels = []
        record_products: set[str] = set()
        for feature in cds:
            label = feature.qualifiers.get("gene", feature.qualifiers.get("product", ["unknown"]))[0]
            gene_labels.append(f"{label}:{int(len(feature.location))}")
            product = feature.qualifiers.get("product", ["unknown"])[0]
            product_counts[product] += 1
            record_products.add(product)
        for product in record_products:
            product_genomes[product] += 1
        tables = sorted({
            str(table)
            for feature in cds
            for table in feature.qualifiers.get("transl_table", ["1"])
        })
        rows.append({
            "accession": record.id,
            "record_name": record.name,
            "description": record.description,
            "length_bp": len(sequence),
            "all_acgt": valid_bases,
            "topology": record.annotations.get("topology", ""),
            "genbank_date": record.annotations.get("date", ""),
            "cds_count": len(cds),
            "cds_ge_150nt": sum(n >= 150 for n in cds_lengths),
            "translated_cds_count": sum("translation" in feature.qualifiers for feature in cds),
            "translation_tables": ";".join(tables),
            "gene_size_labels": ";".join(gene_labels),
        })
        candidate = (
            valid_bases and valid_length
            and sum(n >= 150 and "translation" in feature.qualifiers
                    for n, feature in zip(cds_lengths, cds)) >= 6
        )
        if candidate:
            for product in record_products:
                candidate_product_genomes[product] += 1
            digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            candidate_by_hash.setdefault(digest, {
                "accession": record.id,
                "record_name": record.name,
                "description": record.description,
                "length_bp": len(sequence),
                "cds_count": len(cds),
                "translated_cds_ge_150nt": sum(
                    n >= 150 and "translation" in feature.qualifiers
                    for n, feature in zip(cds_lengths, cds)
                ),
                "translation_tables": ";".join(tables),
                "sha256": digest,
            })

    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise SystemExit(f"accession mismatch: missing={len(missing)}, extra={len(extra)}")
    if len(rows) != len(expected):
        raise SystemExit(f"record count {len(rows)} does not match accession count {len(expected)}")

    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with CANDIDATES.open("w", newline="", encoding="utf-8") as handle:
        fields = list(next(iter(candidate_by_hash.values())))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_by_hash.values())
    summary = {
        "query_manifest": "data/microviridae_query_manifest.txt",
        "accession_file": str(ACCESSIONS),
        "genbank_file": str(GENBANK),
        "inventory_file": str(INVENTORY),
        "min_length_bp": min(lengths),
        "max_length_bp": max(lengths),
        "unique_exact_sequences": len(hashes),
        "exact_duplicate_records": sum(n - 1 for n in hashes.values()),
        "largest_exact_duplicate_group": max(hashes.values()),
        "screened_candidates_after_exact_dedup": len(candidate_by_hash),
        "most_common_cds_products": product_counts.most_common(25),
        "most_common_products_by_genome": product_genomes.most_common(25),
        "most_common_products_in_screened_genomes": candidate_product_genomes.most_common(25),
        "counts": dict(counts),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
