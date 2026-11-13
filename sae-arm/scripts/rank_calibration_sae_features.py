"""Rank SAE features by family using paired calibration-genome contrasts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sae-arm" / "results"
FEATURES = RESULTS / "microviridae_calibration_sae_gene_features_20261112.npz"
CANDIDATES = RESULTS / "microviridae_calibration_sae_feature_candidates_20261112.tsv"
SUMMARY = RESULTS / "microviridae_calibration_sae_feature_ranking_20261112.json"
TOP_PER_FAMILY = 100

FAMILY_NAMES = {
    "MT310288.1|cds01": "major capsid protein",
    "MW697641.1|cds06": "replication initiator protein",
    "MZ089757.1|cds01": "nonstructural protein",
    "MT310270.1|cds06": "internal scaffolding protein",
    "MK249233.1|cds03": "DNA pilot protein",
}


def main() -> None:
    with np.load(FEATURES, allow_pickle=False) as archive:
        matrix = archive["gene_mean_features"].astype(np.float64)
        rows = json.loads(str(archive["row_metadata_json"]))

    if matrix.shape != (150, 32768) or len(rows) != 150:
        raise SystemExit(f"Unexpected calibration feature matrix or metadata: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise SystemExit("Feature matrix contains non-finite values")

    family_ids = sorted({str(row["family_id"]) for row in rows})
    by_genome: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        accession = str(row["accession"])
        family_id = str(row["family_id"])
        if family_id in by_genome[accession]:
            raise SystemExit(f"Duplicate family gene for {accession}: {family_id}")
        by_genome[accession][family_id] = index
    if len(by_genome) != 30 or any(set(group) != set(family_ids) for group in by_genome.values()):
        raise SystemExit("Expected exactly one of each of the five family genes in each of 30 genomes")

    accessions = sorted(by_genome)
    candidate_rows: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    for family_id in family_ids:
        positive = np.stack([matrix[by_genome[acc][family_id]] for acc in accessions])
        control = np.stack([
            np.stack([
                matrix[by_genome[acc][other_id]]
                for other_id in family_ids if other_id != family_id
            ]).mean(axis=0)
            for acc in accessions
        ])
        paired_delta = positive - control
        median_delta = np.median(paired_delta, axis=0)
        mean_delta = paired_delta.mean(axis=0)
        positive_fraction = (paired_delta > 0).mean(axis=0)
        target_active_fraction = (positive > 0).mean(axis=0)
        control_active_fraction = (control > 0).mean(axis=0)
        paired_sd = paired_delta.std(axis=0, ddof=1)
        standardized_effect = mean_delta / np.maximum(paired_sd, 1e-12)

        # Calibration-only rule: prefer features with a positive typical
        # within-genome contrast, then consistency and mean effect.
        order = np.lexsort((
            np.arange(matrix.shape[1]),
            -mean_delta,
            -positive_fraction,
            -median_delta,
        ))
        chosen = int(order[0])
        selected.append({
            "family_id": family_id,
            "family_name": FAMILY_NAMES.get(family_id, "unannotated family"),
            "feature_id": chosen,
            "median_paired_delta": float(median_delta[chosen]),
            "mean_paired_delta": float(mean_delta[chosen]),
            "positive_genome_fraction": float(positive_fraction[chosen]),
            "family_gene_active_fraction": float(target_active_fraction[chosen]),
            "other_family_active_fraction": float(control_active_fraction[chosen]),
        })

        for rank, feature_id in enumerate(order[:TOP_PER_FAMILY], start=1):
            candidate_rows.append({
                "family_id": family_id,
                "family_name": FAMILY_NAMES.get(family_id, "unannotated family"),
                "candidate_rank": rank,
                "feature_id": int(feature_id),
                "median_paired_delta": float(median_delta[feature_id]),
                "mean_paired_delta": float(mean_delta[feature_id]),
                "positive_genome_fraction": float(positive_fraction[feature_id]),
                "family_gene_active_fraction": float(target_active_fraction[feature_id]),
                "other_family_active_fraction": float(control_active_fraction[feature_id]),
                "paired_standardized_effect": float(standardized_effect[feature_id]),
            })

    fields = list(candidate_rows[0])
    with CANDIDATES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary = {
        "status": "PASS",
        "created_on": "2026-11-12",
        "source": FEATURES.name,
        "split_used": "30 calibration genomes only",
        "held_out_test_genomes_read": False,
        "comparison": "For each genome and family, compare that gene's mean SAE activation with the mean across the other four family genes from the same genome.",
        "selection_rule": "Rank each feature by median paired activation difference, then fraction of genomes with positive differences, then mean paired difference. Select the top feature per family for calibration follow-up only.",
        "caveat": "Exploratory feature ranking across 32,768 candidates; these ranks are not significance tests and require validation on the held-out test genomes.",
        "genome_count": len(accessions),
        "family_gene_count": len(rows),
        "candidate_features_per_family": TOP_PER_FAMILY,
        "selected_calibration_candidates": selected,
        "candidate_table": CANDIDATES.name,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "candidate_table": str(CANDIDATES),
        "summary": str(SUMMARY),
        "selected_calibration_candidates": selected,
        "held_out_test_genomes_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
