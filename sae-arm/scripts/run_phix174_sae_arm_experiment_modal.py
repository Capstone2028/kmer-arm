"""Matched three-seed phiX174 comparison: native, temperature, k-mer, and SAE."""

from __future__ import annotations

import json
import time
import hashlib
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
PHIX174 = ROOT / "phix174.fasta"
KMER_TABLE = ROOT / "dependencies" / "microviridae_k346.npz"
CALIBRATION_RANKING = RESULTS / "microviridae_calibration_sae_feature_ranking_20261112.json"
TEST_VALIDATION = RESULTS / "microviridae_test_sae_candidate_validation_20261112.json"
OUTPUT = RESULTS / "phix174_sae_arm_experiment_20261112.json"
MODEL_NAME = "evo2_7b_262k"
SEEDS = (0, 1, 2)
PROMPT_LENGTH = 64
TEMPERATURE = 1.0
TEMPERATURE_CONTROL = 0.7
TOP_K = 4
STEERING_STRENGTH = 0.25
KMER_CANDIDATES = 3
KMER_CHUNK_LENGTH = 512
TIMEOUT_SECONDS = 3600

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3").run_commands(
    "python -m pip install evo2==0.6.0"
)
app = modal.App("kmer-arm-phix174-sae-four-arm-experiment")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=TIMEOUT_SECONDS)
def run_matched_arms(
    prompt: str,
    n_tokens: int,
    seeds: list[int],
    steering_feature_id: int,
    steering_strength: float,
    kmer_table_bytes: bytes,
) -> str:
    import io
    import numpy as np
    import torch
    from evo2 import Evo2
    from huggingface_hub import hf_hub_download

    torch.set_grad_enabled(False)
    model = Evo2(MODEL_NAME)
    base_model = model.model
    base_model.eval()
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    gpu_name = torch.cuda.get_device_name(0)

    checkpoint_path = Path(hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
    ))
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    expected_hash = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"
    if digest.hexdigest() != expected_hash:
        raise AssertionError("SAE checkpoint checksum did not match the inspected file")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = {
        key.removeprefix("_orig_mod.").removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    class TiedBatchTopKSAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(checkpoint["W"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_enc = torch.nn.Parameter(checkpoint["b_enc"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_dec = torch.nn.Parameter(checkpoint["b_dec"].to(device=device, dtype=dtype), requires_grad=False)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            values = torch.relu(x @ self.W + self.b_enc)
            positions = values.numel() // values.shape[-1]
            top = torch.topk(values.flatten(), 64 * positions)
            sparse = torch.zeros_like(values.flatten()).scatter(0, top.indices, top.values)
            return sparse.reshape(values.shape)

        def decode(self, features: torch.Tensor) -> torch.Tensor:
            return features @ self.W.T + self.b_dec

    sae = TiedBatchTopKSAE().eval()
    module_paths: dict[str, torch.nn.Module] = {}

    def collect_modules(module: torch.nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = prefix + name
            module_paths[full_name] = child
            collect_modules(child, full_name + "-")

    collect_modules(base_model)
    if "blocks-26" not in module_paths:
        raise AssertionError("Could not locate the SAE's Evo 2 layer-26 module")
    layer_26 = module_paths["blocks-26"]

    table = np.load(io.BytesIO(kmer_table_bytes))
    ks = [int(k) for k in table["ks"]]
    probabilities = {}
    for k in ks:
        counts = table[f"c{k}"]
        total = counts.sum()
        probabilities[k] = counts / total if total > 0 else counts
    base_lut = np.full(256, -1, dtype=np.int64)
    for index, base in enumerate("ACGT"):
        base_lut[ord(base)] = index
        base_lut[ord(base.lower())] = index

    def kmer_indices(sequence: str, k: int) -> np.ndarray:
        encoded = base_lut[np.frombuffer(sequence.encode(), dtype=np.uint8)]
        n = len(encoded) - k + 1
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        indices = np.zeros(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        for offset in range(k):
            segment = encoded[offset:offset + n]
            valid &= segment >= 0
            indices = indices * 4 + np.where(segment >= 0, segment, 0)
        return indices[valid]

    def paper_score(candidate: str, context: str) -> float:
        if not candidate:
            return 0.0
        total = 0.0
        for k in ks:
            prefix = context[-(k - 1):] if k > 1 and context else ""
            indices = kmer_indices(prefix + candidate, k)
            if len(indices):
                total += probabilities[k][indices].sum()
        return float(total / len(candidate))

    def set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def generate(
        context: str,
        count: int,
        seed: int,
        temperature: float,
        steer: bool = False,
    ) -> tuple[str, float, dict[str, object]]:
        set_seed(seed)
        hook_shapes: dict[tuple[int, ...], int] = {}
        handle = None
        if steer:
            def sae_hook(
                _module: torch.nn.Module,
                _inputs: tuple[object, ...],
                output: object,
            ) -> object:
                tuple_output = isinstance(output, tuple)
                activations = output[0] if tuple_output else output
                if not isinstance(activations, torch.Tensor) or activations.shape[-1] != 4096:
                    raise TypeError("Unexpected layer-26 activation shape during generation")
                shape = tuple(int(dim) for dim in activations.shape)
                hook_shapes[shape] = hook_shapes.get(shape, 0) + 1
                features = sae.encode(activations)
                features = features.clone()
                features[..., steering_feature_id] += steering_strength
                reconstructed = sae.decode(features)
                return (reconstructed, *output[1:]) if tuple_output else reconstructed

            handle = layer_26.register_forward_hook(sae_hook)

        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = model.generate(
                    prompt_seqs=[context],
                    n_tokens=count,
                    temperature=temperature,
                    top_k=TOP_K,
                    cached_generation=True,
                    verbose=0,
                )
            continuation = str(output.sequences[0])
        finally:
            if handle is not None:
                handle.remove()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(continuation) == len(context) + count and continuation.startswith(context):
            continuation = continuation[len(context):]
        if len(continuation) != count:
            raise AssertionError(f"Expected {count} generated bases; got {len(continuation)}")
        shape_summary = {
            "hook_calls": sum(hook_shapes.values()),
            "activation_shapes": [
                {"shape": list(shape), "calls": count_for_shape}
                for shape, count_for_shape in sorted(hook_shapes.items())
            ],
        }
        return continuation, elapsed, shape_summary

    result_rows = []
    for seed in seeds:
        arms: dict[str, dict[str, object]] = {}
        context = prompt

        continuation, elapsed, _ = generate(context, n_tokens, seed, TEMPERATURE)
        arms["native"] = {"sequence": continuation, "seconds": round(elapsed, 3)}

        continuation, elapsed, _ = generate(context, n_tokens, seed, TEMPERATURE_CONTROL)
        arms["temperature_control"] = {
            "temperature": TEMPERATURE_CONTROL,
            "sequence": continuation,
            "seconds": round(elapsed, 3),
        }

        continuation, elapsed, hook_summary = generate(
            context, n_tokens, seed, TEMPERATURE, steer=True,
        )
        arms["sae_steering"] = {
            "feature_id": steering_feature_id,
            "feature_family": "major capsid protein",
            "activation_addition": steering_strength,
            "sequence": continuation,
            "seconds": round(elapsed, 3),
            "hook": hook_summary,
        }

        sequence = context
        rounds = []
        kmer_started = time.perf_counter()
        round_index = 0
        while len(sequence) - len(context) < n_tokens:
            chunk_size = min(KMER_CHUNK_LENGTH, n_tokens - (len(sequence) - len(context)))
            candidate_seeds = [seed * 1_000_003 + round_index * 1009 + j for j in range(KMER_CANDIDATES)]
            candidates = []
            scores = []
            for candidate_seed in candidate_seeds:
                candidate, _, _ = generate(sequence, chunk_size, candidate_seed, TEMPERATURE)
                candidates.append(candidate)
                scores.append(paper_score(candidate, sequence))
            picked = int(np.argmax(scores))
            sequence += candidates[picked]
            rounds.append({
                "round": round_index,
                "candidate_seeds": candidate_seeds,
                "scores": [round(score, 8) for score in scores],
                "picked_candidate": picked,
            })
            round_index += 1
        arms["kmer_selection"] = {
            "score_mode": "paper",
            "candidate_count": KMER_CANDIDATES,
            "chunk_length": KMER_CHUNK_LENGTH,
            "sequence": sequence[len(context):],
            "seconds": round(time.perf_counter() - kmer_started, 3),
            "rounds": rounds,
        }
        result_rows.append({"seed": seed, "arms": arms})

    return json.dumps({
        "model": MODEL_NAME,
        "model_checkpoint_for_all_arms": "evo2_7b_262k",
        "gpu": gpu_name,
        "prompt_length": len(prompt),
        "generated_bases_per_arm": n_tokens,
        "seeds": seeds,
        "arms": result_rows,
        "biological_data_used_for_generation_prompt": "phiX174 first 64 bases",
    })


@app.local_entrypoint()
def main() -> None:
    import numpy as np
    import bio_scoring
    import genome_scoring
    from Bio import SeqIO

    selected = json.loads(CALIBRATION_RANKING.read_text(encoding="utf-8"))["selected_calibration_candidates"]
    validation = json.loads(TEST_VALIDATION.read_text(encoding="utf-8"))
    major_capsid = next(row for row in selected if row["family_name"] == "major capsid protein")
    validation_row = next(
        row for row in validation["candidate_results"]
        if row["family_name"] == "major capsid protein"
    )
    if validation_row["calibration_selected_feature_id"] != int(major_capsid["feature_id"]):
        raise SystemExit("The held-out major-capsid validation does not match the selected SAE feature")

    genome = str(next(SeqIO.parse(PHIX174, "fasta")).seq).upper()
    prompt = genome[:PROMPT_LENGTH]
    n_tokens = len(genome) - PROMPT_LENGTH
    table_bytes = KMER_TABLE.read_bytes()
    raw = json.loads(run_matched_arms.remote(
        prompt,
        n_tokens,
        list(SEEDS),
        int(major_capsid["feature_id"]),
        STEERING_STRENGTH,
        table_bytes,
    ))

    for seed_row in raw["arms"]:
        for arm_name, arm in seed_row["arms"].items():
            continuation = str(arm["sequence"])
            if len(continuation) != n_tokens:
                raise SystemExit(f"Seed {seed_row['seed']} {arm_name}: generated length mismatch")
            assembled = prompt + continuation
            arm["metrics"] = {
                "generated_region": bio_scoring.score_sequence(continuation, organism_coding=True),
                "whole_genome": genome_scoring.score_genome(assembled, prompt_len=PROMPT_LENGTH),
                "vs_real_generated_region": bio_scoring.compare_sequences(
                    continuation, genome[PROMPT_LENGTH:], organism_coding=True,
                ),
            }
            if arm_name != "native":
                native = seed_row["arms"]["native"]["sequence"]
                arm["metrics"]["vs_native_generated_region"] = bio_scoring.compare_sequences(
                    continuation, str(native), organism_coding=True,
                )

    output = {
        "status": "completed",
        "experiment_date": date.today().isoformat(),
        "design": {
            "model": MODEL_NAME,
            "prompt": "first 64 bases of phiX174 NC_001422.1",
            "generated_length": n_tokens,
            "seeds": list(SEEDS),
            "arms": {
                "native": "temperature 1.0, top-k 4, one continuation",
                "temperature_control": f"temperature {TEMPERATURE_CONTROL}, top-k 4",
                "kmer_selection": f"paper score, k={KMER_CANDIDATES} candidates per {KMER_CHUNK_LENGTH}-base chunk",
                "sae_steering": f"feature {major_capsid['feature_id']} (major capsid), +{STEERING_STRENGTH} during layer-26 prefill and cached decode",
            },
            "sae_feature_source": CALIBRATION_RANKING.name,
            "held_out_validation_source": TEST_VALIDATION.name,
            "held_out_major_capsid_feature_fraction_higher": validation_row["fraction_test_genomes_target_higher"],
            "held_out_major_capsid_feature_test_genomes": validation_row["test_genomes_compared"],
            "timeout_seconds": TIMEOUT_SECONDS,
            "estimated_resource_cost_ceiling_usd": 2.40,
            "estimate_note": "One L40S worker, 4 CPU cores, 32 GiB memory, capped at 60 minutes using current published Modal rates; setup may add small overhead.",
        },
        "runs": raw["arms"],
        "real_reference": {
            "accession": "NC_001422.1",
            "length": len(genome),
            "genome_metrics": genome_scoring.score_genome(genome, prompt_len=0),
            "generated_region_metrics": bio_scoring.score_sequence(genome[PROMPT_LENGTH:], organism_coding=True),
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "output": str(OUTPUT),
        "runs": [
            {"seed": row["seed"], "arms": {
                name: {
                    "seconds": details["seconds"],
                    "coding_density": details["metrics"]["generated_region"].get("coding_density"),
                    "gc": details["metrics"]["generated_region"].get("gc"),
                    "n_genes": details["metrics"]["generated_region"].get("n_genes"),
                    "kmer_js_divergence": details["metrics"]["vs_real_generated_region"].get("kmer_js_divergence"),
                    "orf_closed": details["metrics"]["whole_genome"].get("orf_closure", {}).get("n_closed"),
                    "orf_scored": details["metrics"]["whole_genome"].get("orf_closure", {}).get("n_scored"),
                }
                for name, details in row["arms"].items()
            }}
            for row in raw["arms"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
