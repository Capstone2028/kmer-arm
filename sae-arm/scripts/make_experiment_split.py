"""Create a reproducible calibration/test split from the fresh family cohort."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "sae-arm" / "results"
FAMILY_COHORT = RESULTS / "microviridae_five_family_candidates_20261112.tsv"
GENOME_CLUSTERS = RESULTS / "microviridae_genome_clusters_90pct_20261111.tsv"
CANDIDATES = DATA / "microviridae_screened_candidates_20261111.csv"
GENOMES = DATA / "microviridae_screened_genomes_20261111.fna"
PROTEINS = DATA / "microviridae_screened_proteins_20261111.faa"
PROTEIN_TABLE = DATA / "microviridae_screened_proteins_20261111.tsv"

SPLIT = RESULTS / "microviridae_split_20261112.tsv"
MANIFEST = RESULTS / "microviridae_split_manifest_20261112.json"
SPLIT_GENOMES = RESULTS / "microviridae_split_genomes_20261112.fna"
SPLIT_PROTEINS = RESULTS / "microviridae_split_proteins_20261112.faa"
SPLIT_PROTEIN_TABLE = RESULTS / "microviridae_split_proteins_20261112.tsv"
CALIBRATION_FASTA = RESULTS / "microviridae_calibration_genomes_20261112.fna"
TEST_FASTA = RESULTS / "microviridae_test_genomes_20261112.fna"

SEED = 20261112
CALIBRATION_CLUSTERS = 30
TEST_CLUSTERS = 60


def write_fasta(path: Path, records: list) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        SeqIO.write(records, handle, "fasta")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    with FAMILY_COHORT.open(newline="", encoding="utf-8") as handle:
        family_rows = list(csv.DictReader(handle, delimiter="\t"))
    if not family_rows:
        raise SystemExit("Five-family cohort is empty")
    family_fields = [
        name for name in family_rows[0]
        if name not in {"accession", "genome_cluster"}
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in family_rows:
        grouped[row["genome_cluster"]].append(row)

    if len(grouped) < CALIBRATION_CLUSTERS + TEST_CLUSTERS:
        raise SystemExit(
            f"Need {CALIBRATION_CLUSTERS + TEST_CLUSTERS} clusters; found {len(grouped)}"
        )

    with CANDIDATES.open(newline="", encoding="utf-8") as handle:
        candidate_metadata = {row["accession"]: row for row in csv.DictReader(handle)}
    genome_cluster_sizes: Counter[str] = Counter()
    with GENOME_CLUSTERS.open(newline="", encoding="utf-8") as handle:
        for representative, _member in csv.reader(handle, delimiter="\t"):
            genome_cluster_sizes[representative] += 1

    # Prefer the best-annotated family-positive genome within each similarity
    # group; use accession as a stable tie-breaker.
    cluster_representatives: dict[str, dict[str, str]] = {}
    for cluster, rows in grouped.items():
        def quality(row: dict[str, str]) -> tuple[int, int, str]:
            metadata = candidate_metadata[row["accession"]]
            return (
                -int(metadata["translated_cds_ge_150nt"]),
                -int(metadata["cds_count"]),
                row["accession"],
            )

        cluster_representatives[cluster] = min(rows, key=quality)

    shuffled_clusters = sorted(cluster_representatives)
    random.Random(SEED).shuffle(shuffled_clusters)
    calibration_ids = shuffled_clusters[:CALIBRATION_CLUSTERS]
    test_ids = shuffled_clusters[
        CALIBRATION_CLUSTERS:CALIBRATION_CLUSTERS + TEST_CLUSTERS
    ]

    split_rows: list[dict[str, object]] = []
    selected: dict[str, str] = {}
    for split_name, clusters in (("calibration", calibration_ids), ("test", test_ids)):
        for cluster in clusters:
            row = cluster_representatives[cluster]
            accession = row["accession"]
            metadata = candidate_metadata[accession]
            selected[accession] = split_name
            split_rows.append({
                "accession": accession,
                "genome_cluster": cluster,
                "split": split_name,
                "cluster_size_in_screened_dataset": genome_cluster_sizes[cluster],
                "length_bp": metadata["length_bp"],
                "annotated_cds": metadata["cds_count"],
                "translated_cds_ge_150nt": metadata["translated_cds_ge_150nt"],
                "translation_tables": metadata["translation_tables"],
                "family_cluster_ids": ";".join(family_fields),
            })

    row_fields = list(split_rows[0])
    with SPLIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(split_rows)

    selected_genomes = [
        record for record in SeqIO.parse(GENOMES, "fasta") if record.id in selected
    ]
    if len(selected_genomes) != len(selected):
        raise SystemExit("Selected genome FASTA records do not match the split table")
    write_fasta(SPLIT_GENOMES, selected_genomes)
    write_fasta(
        CALIBRATION_FASTA,
        [record for record in selected_genomes if selected[record.id] == "calibration"],
    )
    write_fasta(
        TEST_FASTA,
        [record for record in selected_genomes if selected[record.id] == "test"],
    )

    selected_proteins = [
        record for record in SeqIO.parse(PROTEINS, "fasta")
        if record.id.split("|", 1)[0] in selected
    ]
    write_fasta(SPLIT_PROTEINS, selected_proteins)
    with PROTEIN_TABLE.open(newline="", encoding="utf-8") as source, SPLIT_PROTEIN_TABLE.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            row for row in reader if row["accession"] in selected
        )

    split_counts = Counter(selected.values())
    manifest = {
        "created_on": "2026-11-12",
        "label": "new split; not a reconstruction of the unavailable prior raw split",
        "selection": {
            "source_family_positive_genomes": len(family_rows),
            "available_90pct_genome_clusters": len(grouped),
            "one_genome_per_cluster": True,
            "selection_seed": SEED,
            "calibration_clusters": CALIBRATION_CLUSTERS,
            "test_clusters": TEST_CLUSTERS,
                "cluster_representative_rule": "Best annotated family-positive member; ties use lexicographically smallest accession.",
            "genome_similarity_threshold": "90% nucleotide identity, 80% bidirectional coverage; connected components",
            "protein_family_threshold": "30% amino-acid identity, 80% bidirectional coverage; connected components",
            "five_family_rule": "Five most prevalent protein clusters from this fresh dataset, checked against common product labels.",
        },
        "selected_genomes": len(selected),
        "selected_proteins": len(selected_proteins),
        "split_counts": dict(split_counts),
        "family_cluster_ids": family_fields,
        "outputs": [
            str(path.relative_to(ROOT)) for path in (
                SPLIT, SPLIT_GENOMES, SPLIT_PROTEINS, SPLIT_PROTEIN_TABLE,
                CALIBRATION_FASTA, TEST_FASTA,
            )
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "available_family_positive_clusters": len(grouped),
        "selected_calibration_clusters": split_counts["calibration"],
        "selected_test_clusters": split_counts["test"],
        "selected_genomes": len(selected),
        "selected_proteins": len(selected_proteins),
        "split_file": str(SPLIT),
        "manifest": str(MANIFEST),
    }, indent=2))


if __name__ == "__main__":
    main()
