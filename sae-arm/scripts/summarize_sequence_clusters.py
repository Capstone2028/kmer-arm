"""Turn MMseqs2 cluster memberships into family tables for experiment design."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "sae-arm" / "results"
PROTEINS = DATA / "microviridae_screened_proteins_20261111.tsv"
CLUSTERS = RESULTS / "microviridae_protein_families_30pct_20261111.tsv"
GENOME_CLUSTERS = RESULTS / "microviridae_genome_clusters_90pct_20261111.tsv"
SUMMARY = RESULTS / "microviridae_protein_family_summary_20261111.tsv"
MATRIX = RESULTS / "microviridae_core_family_matrix_20261111.tsv"
FIVE_FAMILIES = RESULTS / "microviridae_five_family_candidates_20261112.tsv"
FIVE_SUMMARY = RESULTS / "microviridae_five_family_summary_20261112.json"
GENOME_SUMMARY = RESULTS / "microviridae_genome_cluster_summary_20261112.json"
MIN_GENOMES = 50


def main() -> None:
    for input_path in (CLUSTERS, GENOME_CLUSTERS):
        if not input_path.exists():
            raise SystemExit(f"Cluster assignments not found: {input_path}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    protein_metadata: dict[str, dict[str, str]] = {}
    genomes: set[str] = set()
    with PROTEINS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            protein_metadata[row["protein_id"]] = row
            genomes.add(row["accession"])

    members: dict[str, list[str]] = defaultdict(list)
    with CLUSTERS.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2:
                raise SystemExit(f"Expected two columns in {CLUSTERS}; got {len(row)}")
            representative, member = row
            if member not in protein_metadata:
                raise SystemExit(f"Cluster output refers to unknown protein: {member}")
            members[representative].append(member)

    clustered_proteins = [member for cluster_members in members.values() for member in cluster_members]
    if len(clustered_proteins) != len(protein_metadata) or set(clustered_proteins) != set(protein_metadata):
        raise SystemExit(
            f"Protein-cluster coverage mismatch: {len(clustered_proteins)} assignments "
            f"for {len(protein_metadata)} proteins"
        )

    family_genomes: dict[str, set[str]] = {}
    summary_rows: list[dict[str, object]] = []
    for representative, proteins in members.items():
        accessions = {protein_metadata[p]["accession"] for p in proteins}
        product_counts: Counter[str] = Counter(
            protein_metadata[p]["product"] or "unannotated" for p in proteins
        )
        family_genomes[representative] = accessions
        summary_rows.append({
            "family_id": representative,
            "protein_count": len(proteins),
            "genome_count": len(accessions),
            "genome_fraction": f"{len(accessions) / len(genomes):.6f}",
            "common_products": " | ".join(
                f"{name} ({count})" for name, count in product_counts.most_common(8)
            ),
        })
    summary_rows.sort(key=lambda row: (-int(row["genome_count"]), str(row["family_id"])))

    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    core_families = [
        row["family_id"] for row in summary_rows if int(row["genome_count"]) >= MIN_GENOMES
    ]
    fields = ["accession", *core_families]
    with MATRIX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for accession in sorted(genomes):
            row: dict[str, object] = {"accession": accession}
            for family in core_families:
                row[family] = int(accession in family_genomes[family])
            writer.writerow(row)

    genome_cluster: dict[str, str] = {}
    genome_cluster_sizes: Counter[str] = Counter()
    with GENOME_CLUSTERS.open(newline="", encoding="utf-8") as handle:
        for representative, member in csv.reader(handle, delimiter="\t"):
            genome_cluster[member] = representative
            genome_cluster.setdefault(representative, representative)
            genome_cluster_sizes[representative] += 1
    if set(genome_cluster) != genomes:
        raise SystemExit(
            f"Genome-cluster accession mismatch: expected {len(genomes)}, "
            f"got {len(genome_cluster)}"
        )
    GENOME_SUMMARY.write_text(json.dumps({
        "candidate_genomes": len(genomes),
        "genome_clusters_at_90pct_identity": len(genome_cluster_sizes),
        "singleton_clusters": sum(size == 1 for size in genome_cluster_sizes.values()),
        "multi_genome_clusters": sum(size > 1 for size in genome_cluster_sizes.values()),
        "largest_cluster_size": max(genome_cluster_sizes.values()),
        "largest_clusters": [
            {"representative": accession, "members": size}
            for accession, size in genome_cluster_sizes.most_common(10)
        ],
    }, indent=2) + "\n", encoding="utf-8")

    # Provisional core-family candidates: five highest-prevalence sequence
    # clusters, reviewed against the dominant annotated product names.
    top_five = summary_rows[:5]
    top_five_ids = [str(row["family_id"]) for row in top_five]
    all_five_genomes = set.intersection(*(family_genomes[f] for f in top_five_ids))
    all_five_groups = {genome_cluster[accession] for accession in all_five_genomes}
    with FIVE_FAMILIES.open("w", newline="", encoding="utf-8") as handle:
        family_fields = ["accession", "genome_cluster", *top_five_ids]
        writer = csv.DictWriter(handle, fieldnames=family_fields, delimiter="\t")
        writer.writeheader()
        for accession in sorted(all_five_genomes):
            row = {"accession": accession, "genome_cluster": genome_cluster[accession]}
            row.update({family: 1 for family in top_five_ids})
            writer.writerow(row)

    family_details = [{
        "family_id": row["family_id"],
        "genome_count": row["genome_count"],
        "common_products": row["common_products"],
    } for row in top_five]
    FIVE_SUMMARY.write_text(json.dumps({
        "selection_note": "Five highest-prevalence protein clusters, cross-checked by common product labels. These are fresh-data candidates, not asserted to be exact family IDs from the unavailable prior split.",
        "candidate_genomes_with_all_five": len(all_five_genomes),
        "90pct_genome_clusters_represented": len(all_five_groups),
        "families": family_details,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Summarized {len(members)} protein clusters over {len(genomes)} genomes")
    print(f"Wrote {SUMMARY}; {len(core_families)} clusters occur in at least {MIN_GENOMES} genomes")
    print(f"Wrote {MATRIX}")
    print(f"Wrote {GENOME_SUMMARY}")
    print(f"All five top families: {len(all_five_genomes)} genomes in {len(all_five_groups)} genome clusters")
    print(f"Wrote {FIVE_FAMILIES} and {FIVE_SUMMARY}")


if __name__ == "__main__":
    main()
