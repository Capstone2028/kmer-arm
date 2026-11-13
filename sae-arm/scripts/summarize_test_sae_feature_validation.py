"""Evaluate calibration-locked SAE features on the held-out test genomes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sae-arm" / "results"
FEATURES = RESULTS / "microviridae_test_sae_selected_features_20261112.npz"
MANIFEST = RESULTS / "microviridae_test_sae_selected_features_manifest_20261112.json"
CALIBRATION_RANKING = RESULTS / "microviridae_calibration_sae_feature_ranking_20261112.json"
SUMMARY = RESULTS / "microviridae_test_sae_candidate_validation_20261112.json"
TABLE = RESULTS / "microviridae_test_sae_candidate_validation_20261112.tsv"
BOOTSTRAP_REPLICATES = 5000

FAMILY_NAMES = {
    "MT310288.1|cds01": "major capsid protein",
    "MW697641.1|cds06": "replication initiator protein",
    "MZ089757.1|cds01": "nonstructural protein",
    "MT310270.1|cds06": "internal scaffolding protein",
    "MK249233.1|cds03": "DNA pilot protein",
}


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[sampled].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    with np.load(FEATURES, allow_pickle=False) as archive:
        values = archive["selected_feature_values"].astype(np.float64)
        feature_ids = archive["selected_feature_ids"].astype(int).tolist()
        rows = json.loads(str(archive["row_metadata_json"]))
    run_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_RANKING.read_text(encoding="utf-8"))

    if values.shape != (300, len(feature_ids)) or len(rows) != 300:
        raise SystemExit(f"Unexpected held-out matrix or metadata: {values.shape}")
    if run_manifest["split"] != "test only" or run_manifest["genome_count"] != 60:
        raise SystemExit("Held-out input manifest does not describe the 60-genome test set")
    if not np.isfinite(values).all():
        raise SystemExit("Held-out feature values contain non-finite entries")

    selected = calibration["selected_calibration_candidates"]
    candidate_by_family = {str(row["family_id"]): int(row["feature_id"]) for row in selected}
    if len(candidate_by_family) != 5 or any(feature not in feature_ids for feature in candidate_by_family.values()):
        raise SystemExit("Held-out matrix does not contain all calibration-locked features")
    column_for_feature = {feature: column for column, feature in enumerate(feature_ids)}

    by_genome: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        accession = str(row["accession"])
        family_id = str(row["family_id"])
        if family_id in by_genome[accession]:
            raise SystemExit(f"Duplicate family-region row for {accession}: {family_id}")
        by_genome[accession][family_id] = index
    families = sorted(FAMILY_NAMES)
    if len(by_genome) != 60 or any(set(group) != set(families) for group in by_genome.values()):
        raise SystemExit("Expected five family-region rows for each of 60 held-out genomes")

    candidate_stats = []
    for family_index, family_id in enumerate(sorted(candidate_by_family)):
        feature_id = candidate_by_family[family_id]
        feature_column = column_for_feature[feature_id]
        paired_deltas: list[float] = []
        target_values: list[float] = []
        control_values: list[float] = []
        excluded_partial_target = 0
        excluded_partial_controls = 0

        for accession, family_rows in by_genome.items():
            target_row_index = family_rows[family_id]
            target_row = rows[target_row_index]
            if target_row["partial_start"] or target_row["partial_end"]:
                excluded_partial_target += 1
                continue
            target_value = float(values[target_row_index, feature_column])
            control_indices = []
            for other_family, other_row_index in family_rows.items():
                if other_family == family_id:
                    continue
                other_row = rows[other_row_index]
                if other_row["partial_start"] or other_row["partial_end"]:
                    excluded_partial_controls += 1
                    continue
                control_indices.append(other_row_index)
            if not control_indices:
                continue
            control_value = float(values[control_indices, feature_column].mean())
            paired_deltas.append(target_value - control_value)
            target_values.append(target_value)
            control_values.append(control_value)

        delta = np.asarray(paired_deltas, dtype=np.float64)
        target_array = np.asarray(target_values, dtype=np.float64)
        control_array = np.asarray(control_values, dtype=np.float64)
        if len(delta) < 50:
            raise SystemExit(f"Too few complete paired genomes for {family_id}: {len(delta)}")
        candidate_stats.append({
            "family_id": family_id,
            "family_name": FAMILY_NAMES[family_id],
            "calibration_selected_feature_id": feature_id,
            "test_genomes_compared": int(len(delta)),
            "excluded_partial_target_genomes": excluded_partial_target,
            "partial_control_regions_skipped": excluded_partial_controls,
            "mean_target_activation": float(target_array.mean()),
            "mean_within_genome_control_activation": float(control_array.mean()),
            "mean_paired_difference": float(delta.mean()),
            "mean_paired_difference_bootstrap_95pct_ci": bootstrap_mean_ci(delta, 20261112 + family_index),
            "median_paired_difference": float(np.median(delta)),
            "fraction_test_genomes_target_higher": float((delta > 0).mean()),
            "fraction_ties": float((delta == 0).mean()),
            "candidate_active_fraction_in_target_genes": float((target_array > 0).mean()),
            "candidate_active_fraction_in_control_genes": float((control_array > 0).mean()),
        })

    cross_family_means = []
    for feature_id in feature_ids:
        feature_column = column_for_feature[feature_id]
        per_family = {}
        for family_id in families:
            family_gene_values = [
                float(values[row_index, feature_column])
                for family_rows in by_genome.values()
                if not (rows[(row_index := family_rows[family_id])]["partial_start"]
                        or rows[row_index]["partial_end"])
            ]
            per_family[family_id] = {
                "family_name": FAMILY_NAMES[family_id],
                "n_complete_genes": len(family_gene_values),
                "mean_activation": float(np.mean(family_gene_values)),
                "active_fraction": float(np.mean(np.asarray(family_gene_values) > 0)),
            }
        cross_family_means.append({"feature_id": feature_id, "by_gene_family": per_family})

    summary = {
        "status": "PASS",
        "created_on": "2026-11-12",
        "model": "evo2_7b_262k",
        "split": "held-out test only",
        "calibration_selection_source": CALIBRATION_RANKING.name,
        "test_features_did_not_change_candidate_selection": True,
        "genome_count": len(by_genome),
        "family_gene_count": len(rows),
        "partial_gene_regions_excluded_from_primary_comparisons": run_manifest["partial_family_gene_count"],
        "comparison": "Within each test genome, the calibration-selected feature's mean activation on its target family gene is compared with its mean across the other complete family genes.",
        "uncertainty": f"Percentile bootstrap 95% intervals for the mean paired difference, {BOOTSTRAP_REPLICATES} resamples of test genomes; descriptive, not a multiple-testing-adjusted significance test.",
        "candidate_results": candidate_stats,
        "candidate_feature_cross_family_means": cross_family_means,
        "duplicate_calibration_candidate_ids": sorted(
            feature for feature in set(candidate_by_family.values())
            if list(candidate_by_family.values()).count(feature) > 1
        ),
        "table": TABLE.name,
    }
    fields = list(candidate_stats[0])
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(candidate_stats)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "summary": str(SUMMARY),
        "table": str(TABLE),
        "candidate_results": candidate_stats,
        "duplicate_calibration_candidate_ids": summary["duplicate_calibration_candidate_ids"],
        "test_features_did_not_change_candidate_selection": True,
    }, indent=2))


if __name__ == "__main__":
    main()
