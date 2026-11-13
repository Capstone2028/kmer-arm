"""Tiny CPU-only Modal test for the MMseqs2 clustering commands."""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
REPORT = RESULTS / "modal_mmseqs_smoke_20261112.json"
image = modal.Image.debian_slim(python_version="3.11").apt_install("mmseqs2")
app = modal.App("kmer-arm-mmseqs-smoke-test")


def make_variants(alphabet: str, length: int, changes: int, seed: int) -> tuple[str, str, str]:
    rng = random.Random(seed)
    first = "".join(rng.choice(alphabet) for _ in range(length))
    second = list(first)
    for index in rng.sample(range(length), changes):
        alternatives = alphabet.replace(first[index], "")
        second[index] = rng.choice(alternatives)
    unrelated = "".join(rng.choice(alphabet) for _ in range(length))
    return first, "".join(second), unrelated


@app.function(image=image, cpu=1, memory=2048, timeout=180)
def run_smoke_test(dna_fasta: str, protein_fasta: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mmseqs-smoke-") as directory:
        root = Path(directory)
        dna_path = root / "genomes.fna"
        protein_path = root / "proteins.faa"
        dna_path.write_text(dna_fasta, encoding="ascii")
        protein_path.write_text(protein_fasta, encoding="ascii")
        version = subprocess.run(
            ["mmseqs", "version"], check=True, capture_output=True, text=True
        ).stdout.strip()

        def check_pairwise_cluster(
            fasta: Path, prefix: Path, database_type: int, min_identity: str,
            expected_same: tuple[str, str], expected_separate: tuple[str, str]
        ) -> int:
            temp = root / f"tmp-{prefix.name}"
            command = [
                "mmseqs", "easy-cluster", str(fasta), str(prefix), str(temp),
                "--min-seq-id", min_identity, "-c", "0.80",
                "--cov-mode", "0", "--cluster-mode", "1",
                "--alignment-mode", "3", "--threads", "1",
                "--dbtype", str(database_type),
            ]
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(
                    f"MMseqs2 failed ({completed.returncode}) for {database_type=}:\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )
            assignments: dict[str, str] = {}
            for line in Path(str(prefix) + "_cluster.tsv").read_text(encoding="utf-8").splitlines():
                representative, member = line.split("\t")
                assignments[member] = representative
                assignments.setdefault(representative, representative)
            if assignments.get(expected_same[0]) != assignments.get(expected_same[1]):
                raise RuntimeError(f"Expected similar sequences to cluster: {expected_same}")
            if assignments.get(expected_separate[0]) == assignments.get(expected_separate[1]):
                raise RuntimeError(f"Expected unrelated sequences to remain separate: {expected_separate}")
            return len(set(assignments.values()))

        dna_clusters = check_pairwise_cluster(
            dna_path, root / "dna-result", 2, "0.90", ("dna_a", "dna_b"), ("dna_a", "dna_c")
        )
        protein_clusters = check_pairwise_cluster(
            protein_path, root / "protein-result", 1, "0.30",
            ("protein_a", "protein_b"), ("protein_a", "protein_c")
        )
        return {
            "mmseqs_version": version,
            "dna_cluster_count": dna_clusters,
            "protein_cluster_count": protein_clusters,
            "status": "PASS",
        }


@app.local_entrypoint()
def main() -> None:
    dna = make_variants("ACGT", 1200, 18, 20261112)
    amino_acids = make_variants("ACDEFGHIKLMNPQRSTVWY", 300, 12, 20261113)
    dna_fasta = "".join(
        f">dna_{label}\n{sequence}\n"
        for label, sequence in zip(("a", "b", "c"), dna)
    )
    protein_fasta = "".join(
        f">protein_{label}\n{sequence}\n"
        for label, sequence in zip(("a", "b", "c"), amino_acids)
    )
    result = run_smoke_test.remote(dna_fasta, protein_fasta)
    result.update({
        "date": date.today().isoformat(),
        "gpu_used": False,
        "compute_limit_seconds": 180,
        "estimated_runtime_resource_cost_usd": 0.004,
        "note": "Tiny synthetic inputs; this validates cloud/tool setup, not the biological data or final thresholds.",
    })
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
