"""Unseeded four-arm generation from the team's 25 Reliable25 G2 prompts."""

from __future__ import annotations

import json
import math
import time
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
INPUTS = ROOT / "sae-arm" / "inputs"
RESULTS = ROOT / "sae-arm" / "results"
MANIFEST = INPUTS / "reliable25_300bp_prompt_manifest.json"
REFERENCES = INPUTS / "reliable25_300bp_oriented_references.fna"
KMER_TABLE = ROOT / "dependencies" / "microviridae_k346.npz"
CALIBRATION_RANKING = RESULTS / "microviridae_calibration_sae_feature_ranking_20261112.json"
TEST_VALIDATION = RESULTS / "microviridae_test_sae_candidate_validation_20261112.json"
DEFAULT_OUTPUT = RESULTS / "reliable25_300bp_sae_four_arm_experiment_20261113.json"

MODEL_NAME = "evo2_7b"
TEMPERATURE = 1.0
TEMPERATURE_CONTROL = 0.7
TOP_K = 4
STEERING_STRENGTH = 0.25
KMER_CANDIDATES = 3
KMER_CHUNK_LENGTH = 512
KMER_CONTEXT_LIMIT = 3000
PROMPTS_PER_BATCH = 2
TIMEOUT_SECONDS = 2400
AUTONOMOUS_TIMEOUT_SECONDS = 5 * 60 * 60
ESTIMATED_COST_CEILING_PER_BATCH_USD = 1.60
EXPECTED_PROMPTS = 25

HF_CACHE_PATH = "/cache/huggingface"
hf_cache = modal.Volume.from_name(
    "kmer-arm-huggingface-cache", create_if_missing=True, version=2,
)
results_volume = modal.Volume.from_name(
    "kmer-arm-reliable25-results", create_if_missing=True, version=2,
)
image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3").run_commands(
    "python -m pip install evo2==0.6.0"
).run_commands(
    "python -m pip install pyrodigal==3.7.1"
).env({
    "HF_HOME": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONPATH": "/root",
}).add_local_file(
    str(ROOT / "bio_scoring.py"), remote_path="/root/bio_scoring.py",
)
app = modal.App("kmer-arm-reliable25-sae-four-arm-experiment")


@app.function(
    image=image, cpu=4, memory=16384, timeout=3600,
    volumes={"/cache": hf_cache},
)
def prepare_model_cache() -> None:
    """Download model/SAE weights on CPU and persist them before GPU work."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=f"arcinstitute/{MODEL_NAME}", filename=f"{MODEL_NAME}.pt",
    )
    hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
    )
    hf_cache.commit()


@app.function(
    image=image, gpu="L40S", cpu=4, memory=32768, timeout=AUTONOMOUS_TIMEOUT_SECONDS,
    volumes={"/cache": hf_cache, "/results": results_volume},
)
def run_prompt_batch(
    records: list[dict[str, object]],
    steering_feature_id: int,
    steering_strength: float,
    kmer_table_bytes: bytes,
    prior_completed_json: str = "[]",
    output_name: str = DEFAULT_OUTPUT.name,
    autonomous: bool = False,
) -> str:
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import io
    import hashlib
    import gc
    import numpy as np
    import torch
    from evo2 import Evo2
    from huggingface_hub import hf_hub_download
    import bio_scoring

    max_records = EXPECTED_PROMPTS if autonomous else PROMPTS_PER_BATCH
    if not 1 <= len(records) <= max_records:
        raise ValueError(f"Call must contain 1..{max_records} prompts")

    torch.set_grad_enabled(False)
    model = Evo2(MODEL_NAME)
    # Keep the large Hugging Face checkpoints between Modal function calls.
    hf_cache.commit()
    base_model = model.model
    base_model.eval()
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype

    checkpoint_path = Path(hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
    ))
    hf_cache.commit()
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
    modules: dict[str, torch.nn.Module] = {}

    def collect_modules(module: torch.nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = prefix + name
            modules[full_name] = child
            collect_modules(child, full_name + "-")

    collect_modules(base_model)
    if "blocks-26" not in modules:
        raise AssertionError("Could not locate the SAE's Evo 2 layer-26 module")
    layer_26 = modules["blocks-26"]

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

    def kmer_score(candidate: str, context: str) -> float:
        if not candidate:
            return 0.0
        total = 0.0
        for k in ks:
            prefix = context[-(k - 1):] if k > 1 and context else ""
            indices = kmer_indices(prefix + candidate, k)
            if len(indices):
                total += probabilities[k][indices].sum()
        return float(total / len(candidate))

    def generate(context: str, count: int, temperature: float, steer: bool = False):
        hook_shapes: dict[tuple[int, ...], int] = {}
        handle = None
        if steer:
            def sae_hook(_module, _inputs, output):
                tuple_output = isinstance(output, tuple)
                activations = output[0] if tuple_output else output
                if not isinstance(activations, torch.Tensor) or activations.shape[-1] != 4096:
                    raise TypeError("Unexpected layer-26 activation shape during generation")
                shape = tuple(int(dim) for dim in activations.shape)
                hook_shapes[shape] = hook_shapes.get(shape, 0) + 1
                features = sae.encode(activations).clone()
                features[..., steering_feature_id] += steering_strength
                reconstructed = sae.decode(features)
                return (reconstructed, *output[1:]) if tuple_output else reconstructed

            handle = layer_26.register_forward_hook(sae_hook)

        torch.cuda.synchronize()
        started = time.perf_counter()
        output = None
        try:
            with torch.inference_mode():
                output = model.generate(
                    prompt_seqs=[context], n_tokens=count, temperature=temperature,
                    top_k=TOP_K, cached_generation=True, verbose=0,
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
        del output
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"generated={count} context={len(context)} sae={steer} "
            f"elapsed_s={elapsed:.1f} gpu_free_gib={free_bytes / 1024**3:.1f}/"
            f"{total_bytes / 1024**3:.1f}",
            flush=True,
        )
        return continuation, elapsed, {
            "hook_calls": sum(hook_shapes.values()),
            "activation_shapes": [
                {"shape": list(shape), "calls": calls}
                for shape, calls in sorted(hook_shapes.items())
            ],
        }

    result_rows = []
    if autonomous:
        results_volume.reload()
        result_path = Path("/results") / output_name
        if result_path.exists():
            completed = list(json.loads(result_path.read_text(encoding="utf-8")).get("runs", []))
        else:
            completed = list(json.loads(prior_completed_json))
    else:
        completed = []

    def checkpoint_autonomous_results() -> None:
        if not autonomous:
            return
        completed.sort(key=lambda row: int(row["prompt_index"]))
        result = {
            "status": "completed" if len(completed) == EXPECTED_PROMPTS else "partial",
            "created_on": date.today().isoformat(),
            "model": MODEL_NAME,
            "model_context_length_bp": 1_000_000,
            "sae_feature_validation_model": "evo2_7b_262k",
            "sae_feature_transfer_to_generation_model": "not revalidated",
            "prompt_file": "prompts_reliable25_300bp.fa",
            "target_prompt_count": EXPECTED_PROMPTS,
            "requested_prompt_count": EXPECTED_PROMPTS,
            "prompt_length_bp": 300,
            "one_unseeded_output_per_prompt_per_arm": True,
            "random_seed_override": None,
            "arms": {
                "native": f"temperature {TEMPERATURE}, top-k {TOP_K}",
                "temperature_control": f"temperature {TEMPERATURE_CONTROL}, top-k {TOP_K}",
                "kmer_selection": (
                    f"{KMER_CANDIDATES} candidates per {KMER_CHUNK_LENGTH}-base chunk; "
                    f"generation context capped at {KMER_CONTEXT_LIMIT} bp"
                ),
                "sae_steering": f"feature {steering_feature_id}, +{steering_strength} at layer 26",
            },
            "generation_target": "Continue each prompt to its source genome's length.",
            "gpu_timeout_seconds_per_autonomous_call": AUTONOMOUS_TIMEOUT_SECONDS,
            "estimated_resource_cost_ceiling_per_batch_usd": ESTIMATED_COST_CEILING_PER_BATCH_USD,
            "estimated_batch_count_for_25_prompts": math.ceil(25 / PROMPTS_PER_BATCH),
            "estimated_total_resource_ceiling_usd_for_25_prompts": round(
                math.ceil(25 / PROMPTS_PER_BATCH) * ESTIMATED_COST_CEILING_PER_BATCH_USD, 2,
            ),
            "runs": completed,
        }
        temp_path = result_path.with_suffix(result_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(result_path)
        results_volume.commit()

    for record in records:
        accession = str(record["accession"])
        prompt_index = int(record["prompt_index"])
        if autonomous and any(int(row["prompt_index"]) == prompt_index for row in completed):
            print(f"skipping already checkpointed {accession}", flush=True)
            continue
        context = str(record["prompt_sequence"])
        genome_length = int(record["genome_length_bp"])
        n_tokens = genome_length - len(context)
        if len(context) != 300 or n_tokens <= 0:
            raise AssertionError(f"{accession}: invalid prompt or continuation length")
        print(
            f"starting {accession}: prompt={len(context)} bp, "
            f"continuation={n_tokens} bp (no fixed random seed)",
            flush=True,
        )

        arms: dict[str, dict[str, object]] = {}
        continuation, elapsed, _ = generate(context, n_tokens, TEMPERATURE)
        arms["native"] = {"sequence": continuation, "seconds": round(elapsed, 3)}

        continuation, elapsed, _ = generate(context, n_tokens, TEMPERATURE_CONTROL)
        arms["temperature_control"] = {
            "temperature": TEMPERATURE_CONTROL,
            "sequence": continuation,
            "seconds": round(elapsed, 3),
        }

        continuation, elapsed, hook = generate(context, n_tokens, TEMPERATURE, steer=True)
        arms["sae_steering"] = {
            "feature_id": steering_feature_id,
            "feature_family": "major capsid protein (provisional association)",
            "activation_addition": steering_strength,
            "sequence": continuation,
            "seconds": round(elapsed, 3),
            "hook": hook,
        }

        sequence = context
        rounds = []
        kmer_started = time.perf_counter()
        round_index = 0
        while len(sequence) - len(context) < n_tokens:
            chunk_size = min(KMER_CHUNK_LENGTH, n_tokens - (len(sequence) - len(context)))
            candidates = []
            scores = []
            generation_context = sequence[-KMER_CONTEXT_LIMIT:]
            for _ in range(KMER_CANDIDATES):
                candidate, _, _ = generate(generation_context, chunk_size, TEMPERATURE)
                candidates.append(candidate)
                scores.append(kmer_score(candidate, sequence))
            picked = int(np.argmax(scores))
            sequence += candidates[picked]
            rounds.append({
                "round": round_index,
                "generation_context_length_bp": len(generation_context),
                "scores": [round(score, 8) for score in scores],
                "picked_candidate": picked,
            })
            round_index += 1
        arms["kmer_selection"] = {
            "candidate_count": KMER_CANDIDATES,
            "chunk_length": KMER_CHUNK_LENGTH,
            "generation_context_limit_bp": KMER_CONTEXT_LIMIT,
            "sequence": sequence[len(context):],
            "seconds": round(time.perf_counter() - kmer_started, 3),
            "rounds": rounds,
        }
        result_rows.append({
            "prompt_index": prompt_index,
            "accession": accession,
            "prompt_length_bp": len(context),
            "generated_bases_per_arm": n_tokens,
            "arms": arms,
        })
        if autonomous:
            remote_row = result_rows[-1]
            reference = str(record["reference_sequence"])
            real_continuation = reference[len(context):]
            if len(real_continuation) != n_tokens:
                raise AssertionError(f"{accession}: reference continuation length mismatch")
            for arm_name, arm in remote_row["arms"].items():
                generated_sequence = str(arm["sequence"])
                if len(generated_sequence) != len(real_continuation):
                    raise AssertionError(f"{accession} {arm_name}: output length mismatch")
                arm["metrics"] = {
                    "generated_region": bio_scoring.score_sequence(
                        generated_sequence, organism_coding=True,
                    ),
                    "vs_real_generated_region": bio_scoring.compare_sequences(
                        generated_sequence, real_continuation, organism_coding=True,
                    ),
                }
                if arm_name != "native":
                    arm["metrics"]["vs_native_generated_region"] = bio_scoring.compare_sequences(
                        generated_sequence,
                        str(remote_row["arms"]["native"]["sequence"]),
                        organism_coding=True,
                    )
            remote_row["prompt_start_in_source_genome_0based"] = int(
                record["prompt_start_in_source_genome_0based"]
            )
            completed.append(remote_row)
            checkpoint_autonomous_results()

    if autonomous:
        return json.dumps({
            "status": "completed" if len(completed) == EXPECTED_PROMPTS else "partial",
            "completed_prompts": len(completed),
            "output_volume": "kmer-arm-reliable25-results",
            "output_path": f"/results/{output_name}",
        })
    return json.dumps({"gpu": torch.cuda.get_device_name(0), "prompts": result_rows})


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if name is not None:
                records[name.split()[0]] = "".join(chunks).upper()
            name, chunks = line[1:], []
        elif line:
            chunks.append(line)
    if name is not None:
        records[name.split()[0]] = "".join(chunks).upper()
    return records


@app.local_entrypoint()
def main(
    limit_prompts: int = 0,
    start_index: int = 0,
    output_name: str = DEFAULT_OUTPUT.name,
    autonomous: bool = False,
) -> None:
    import bio_scoring

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("prompt_count") != 25:
        raise SystemExit("Reliable25 input preparation has not passed")
    references = read_fasta(REFERENCES)
    ranking = json.loads(CALIBRATION_RANKING.read_text(encoding="utf-8"))
    validation = json.loads(TEST_VALIDATION.read_text(encoding="utf-8"))
    major_capsid = next(
        row for row in ranking["selected_calibration_candidates"]
        if row["family_name"] == "major capsid protein"
    )
    validation_row = next(
        row for row in validation["candidate_results"]
        if row["family_name"] == "major capsid protein"
    )
    feature_id = int(major_capsid["feature_id"])
    if validation_row["calibration_selected_feature_id"] != feature_id:
        raise SystemExit("Held-out major-capsid feature does not match calibration selection")

    prompt_rows = manifest["prompts"][start_index:]
    if limit_prompts > 0:
        prompt_rows = prompt_rows[:limit_prompts]
    if not prompt_rows:
        raise SystemExit("No prompts selected")

    output_path = RESULTS / output_name
    RESULTS.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    if output_path.exists() and start_index == 0:
        raise SystemExit(f"Output already exists; choose another output_name: {output_path}")
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        completed = list(prior.get("runs", []))

    payloads = []
    for row in prompt_rows:
        accession = str(row["accession"])
        reference_key = next(
            (key for key in references if key.split()[0] == accession), None
        )
        if reference_key is None:
            raise SystemExit(f"Oriented reference missing for {accession}")
        reference = references[reference_key]
        prompt_length = int(row["prompt_length_bp"])
        if len(reference) != int(row["genome_length_bp"]):
            raise SystemExit(f"{accession}: reference length disagrees with manifest")
        prompt = reference[:prompt_length]
        if len(prompt) != 300:
            raise SystemExit(f"{accession}: expected a 300-bp prompt")
        payloads.append({
            "prompt_index": int(row["prompt_index"]),
            "accession": accession,
            "prompt_sequence": prompt,
            "genome_length_bp": len(reference),
            "reference_sequence": reference,
            "prompt_start_in_source_genome_0based": int(
                row["prompt_start_in_source_genome_0based"]
            ),
        })

    table_bytes = KMER_TABLE.read_bytes()
    if autonomous:
        if limit_prompts:
            raise SystemExit("Do not combine autonomous mode with limit_prompts")
        if len(completed) != start_index:
            raise SystemExit(
                f"Autonomous handoff mismatch: start_index={start_index}, "
                f"saved prompts={len(completed)}"
            )
        call = run_prompt_batch.spawn(
            payloads,
            feature_id,
            STEERING_STRENGTH,
            table_bytes,
            json.dumps(completed),
            output_name,
            True,
        )
        print(json.dumps({
            "submitted": True,
            "function_call_id": call.object_id,
            "remaining_prompts": len(payloads),
            "completed_before_handoff": len(completed),
            "persistent_volume": "kmer-arm-reliable25-results",
            "persistent_output_path": f"/results/{output_name}",
        }, indent=2))
        return

    print("Preparing model cache on CPU before allocating a GPU", flush=True)
    prepare_model_cache.remote()
    batches = [
        payloads[i:i + PROMPTS_PER_BATCH]
        for i in range(0, len(payloads), PROMPTS_PER_BATCH)
    ]
    for batch_num, batch in enumerate(batches, start=1):

        raw = json.loads(run_prompt_batch.remote(
            batch, feature_id, STEERING_STRENGTH, table_bytes,
        ))
        for remote_row in raw["prompts"]:
            accession = remote_row["accession"]
            ref_row = next(row for row in batch if row["accession"] == accession)
            reference = str(ref_row["reference_sequence"])
            real_continuation = reference[300:]
            if len(real_continuation) != remote_row["generated_bases_per_arm"]:
                raise SystemExit(f"{accession}: generated and reference lengths differ")
            for arm_name, arm in remote_row["arms"].items():
                sequence = str(arm["sequence"])
                if len(sequence) != len(real_continuation):
                    raise SystemExit(f"{accession} {arm_name}: output length mismatch")
                arm["metrics"] = {
                    "generated_region": bio_scoring.score_sequence(sequence, organism_coding=True),
                    "vs_real_generated_region": bio_scoring.compare_sequences(
                        sequence, real_continuation, organism_coding=True,
                    ),
                }
                if arm_name != "native":
                    arm["metrics"]["vs_native_generated_region"] = bio_scoring.compare_sequences(
                        sequence,
                        str(remote_row["arms"]["native"]["sequence"]),
                        organism_coding=True,
                    )
            remote_row["prompt_start_in_source_genome_0based"] = ref_row[
                "prompt_start_in_source_genome_0based"
            ]
        completed.extend(raw["prompts"])
        completed.sort(key=lambda row: int(row["prompt_index"]))

        result = {
            "status": "completed" if len(completed) == EXPECTED_PROMPTS else "partial",
            "created_on": date.today().isoformat(),
            "model": MODEL_NAME,
            "model_context_length_bp": 1_000_000,
            "sae_feature_validation_model": "evo2_7b_262k",
            "sae_feature_transfer_to_generation_model": "not revalidated",
            "prompt_file": "prompts_reliable25_300bp.fa",
            "target_prompt_count": EXPECTED_PROMPTS,
            "requested_prompt_count": len(prompt_rows),
            "prompt_length_bp": 300,
            "one_unseeded_output_per_prompt_per_arm": True,
            "random_seed_override": None,
            "arms": {
                "native": f"temperature {TEMPERATURE}, top-k {TOP_K}",
                "temperature_control": f"temperature {TEMPERATURE_CONTROL}, top-k {TOP_K}",
                "kmer_selection": (
                    f"{KMER_CANDIDATES} candidates per {KMER_CHUNK_LENGTH}-base chunk; "
                    f"generation context capped at {KMER_CONTEXT_LIMIT} bp"
                ),
                "sae_steering": f"feature {feature_id}, +{STEERING_STRENGTH} at layer 26",
            },
            "generation_target": "Continue each prompt to its source genome's length.",
            "gpu_timeout_seconds_per_batch": TIMEOUT_SECONDS,
            "estimated_resource_cost_ceiling_per_batch_usd": ESTIMATED_COST_CEILING_PER_BATCH_USD,
            "estimated_batch_count_for_25_prompts": math.ceil(25 / PROMPTS_PER_BATCH),
            "estimated_total_resource_ceiling_usd_for_25_prompts": round(
                math.ceil(25 / PROMPTS_PER_BATCH) * ESTIMATED_COST_CEILING_PER_BATCH_USD, 2,
            ),
            "runs": completed,
        }
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "batch": batch_num,
            "batch_count": len(batches),
            "completed_prompts": len(completed),
            "output": str(output_path),
        }, indent=2))


if __name__ == "__main__":
    main()
