"""Run CPU-only genome and protein-family clustering on Modal."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
DATA = ROOT / "data"
RESULTS = ROOT / "sae-arm" / "results"
GENOMES = DATA / "microviridae_screened_genomes_20261111.fna"
PROTEINS = DATA / "microviridae_screened_proteins_20261111.faa"
GENOME_OUT = RESULTS / "microviridae_genome_clusters_90pct_20261111.tsv"
PROTEIN_OUT = RESULTS / "microviridae_protein_families_30pct_20261111.tsv"
MANIFEST_OUT = RESULTS / "microviridae_clustering_manifest_20261111.json"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("mmseqs2")
    .add_local_file(str(GENOMES), "/input/genomes.fna")
    .add_local_file(str(PROTEINS), "/input/proteins.faa")
)
app = modal.App("kmer-arm-microviridae-clustering")


@app.function(image=image, cpu=4, memory=8192, timeout=600)
def cluster_sequences() -> dict[str, str]:
    import os

    os.makedirs("/tmp/mmseqs-work", exist_ok=True)

    def run_cluster(input_path: str, output_prefix: str, database_type: int, min_identity: str) -> str:
        temporary = tempfile.mkdtemp(prefix="mmseqs-")
        command = [
            "mmseqs", "easy-cluster", input_path, output_prefix, temporary,
            "--min-seq-id", min_identity,
            "-c", "0.80",
            "--cov-mode", "0",
            "--cluster-mode", "1",
            "--alignment-mode", "3",
            "--threads", "4",
            "--dbtype", str(database_type),
        ]
        subprocess.run(command, check=True, text=True)
        return Path(output_prefix + "_cluster.tsv").read_text(encoding="utf-8")

    version = subprocess.run(
        ["mmseqs", "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    genome_clusters = run_cluster(
        "/input/genomes.fna", "/tmp/mmseqs-work/genomes90", 2, "0.90"
    )
    protein_families = run_cluster(
        "/input/proteins.faa", "/tmp/mmseqs-work/proteins30", 1, "0.30"
    )
    return {
        "mmseqs_version": version,
        "genome_clusters_tsv": genome_clusters,
        "protein_families_tsv": protein_families,
    }


@app.local_entrypoint()
def main() -> None:
    if not GENOMES.exists() or not PROTEINS.exists():
        raise SystemExit("First run scripts/export_candidate_proteins.py to create inputs.")

    result = cluster_sequences.remote()
    RESULTS.mkdir(parents=True, exist_ok=True)
    GENOME_OUT.write_text(result["genome_clusters_tsv"], encoding="utf-8")
    PROTEIN_OUT.write_text(result["protein_families_tsv"], encoding="utf-8")
    manifest = {
        "retrieved_on": "2026-11-11",
        "clustering_run_on": date.today().isoformat(),
        "tool": "MMseqs2",
        "version": result["mmseqs_version"],
        "genome_cluster_parameters": {
            "sequence_type": "nucleotide",
            "minimum_identity": 0.90,
            "minimum_coverage": 0.80,
            "coverage_mode": "both sequences",
            "cluster_mode": "connected components",
            "alignment_identity_mode": "exact residue identity",
        },
        "protein_family_parameters": {
            "sequence_type": "amino acid",
            "minimum_identity": 0.30,
            "minimum_coverage": 0.80,
            "coverage_mode": "both sequences",
            "cluster_mode": "connected components",
            "alignment_identity_mode": "exact residue identity",
        },
        "compute": {
            "gpu": False,
            "cpu_cores": 4,
            "memory_mib": 8192,
            "timeout_seconds": 600,
            "estimated_resource_cost_ceiling_usd": 0.05,
            "estimate_note": "Compute-only estimate; actual invoice and image-build treatment may vary.",
        },
        "inputs": {
            "genomes_fasta": str(GENOMES.relative_to(ROOT)),
            "proteins_fasta": str(PROTEINS.relative_to(ROOT)),
            "candidate_genomes": 2549,
            "protein_sequences": 19049,
        },
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {GENOME_OUT}")
    print(f"Wrote {PROTEIN_OUT}")
    print(f"Wrote {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
