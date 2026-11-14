"""Bounded matched SAE/native comparison across Evo 2 context models and prompts."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "sae-arm" / "results"
PHIX174 = ROOT / "phix174.fasta"
OUT = RESULTS / "sae_context_prompt_factorial_20261114.json"
MODELS = ("evo2_7b_262k", "evo2_7b")
PROMPT_LENGTHS = (64, 300)
SEEDS = (0, 1, 2)
FEATURE_ID = 16115
STEERING_STRENGTH = 0.25
TOP_K = 4
GPU_TIMEOUT_SECONDS = 13_000

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3").run_commands(
    "python -m pip install evo2==0.6.0 pyrodigal==3.7.1"
).add_local_file(str(ROOT / "bio_scoring.py"), remote_path="/root/bio_scoring.py")
app = modal.App("sae-context-prompt-factorial")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=GPU_TIMEOUT_SECONDS)
def run_factorial(reference: str) -> str:
    import torch
    from evo2 import Evo2
    from huggingface_hub import hf_hub_download
    import sys
    sys.path.insert(0, "/root")
    import bio_scoring

    torch.set_grad_enabled(False)
    checkpoint_path = Path(hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
    ))
    h = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    expected = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"
    if h != expected:
        raise RuntimeError("SAE checkpoint checksum mismatch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod.").removeprefix("module."): v for k, v in state.items()}
    runs = []
    model_summaries = {}

    class SAE(torch.nn.Module):
        def __init__(self, dtype, device):
            super().__init__()
            self.W = torch.nn.Parameter(state["W"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_enc = torch.nn.Parameter(state["b_enc"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_dec = torch.nn.Parameter(state["b_dec"].to(device=device, dtype=dtype), requires_grad=False)
        def encode(self, x):
            values = torch.relu(x @ self.W + self.b_enc)
            positions = values.numel() // values.shape[-1]
            top = torch.topk(values.flatten(), 64 * positions)
            return torch.zeros_like(values.flatten()).scatter(0, top.indices, top.values).reshape(values.shape)
        def decode(self, f):
            return f @ self.W.T + self.b_dec

    for model_name in MODELS:
        model = Evo2(model_name)
        base = model.model
        base.eval()
        device, dtype = next(base.parameters()).device, next(base.parameters()).dtype
        sae = SAE(dtype, device).eval()
        modules = {}
        def collect(mod, prefix=""):
            for name, child in mod.named_children():
                key = prefix + name
                modules[key] = child
                collect(child, key + "-")
        collect(base)
        layer = modules["blocks-26"]

        def set_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        def activation_probe(prompt):
            captured = {}
            def probe(_m, _i, output):
                x = output[0] if isinstance(output, tuple) else output
                f = sae.encode(x)
                captured["feature_mean"] = float(f[..., FEATURE_ID].float().mean().item())
                captured["feature_nonzero_fraction"] = float((f[..., FEATURE_ID] > 0).float().mean().item())
                return output
            handle = layer.register_forward_hook(probe)
            try:
                with torch.inference_mode():
                    model.generate(prompt_seqs=[prompt], n_tokens=1, temperature=1.0, top_k=TOP_K,
                                   cached_generation=True, verbose=0)
            finally:
                handle.remove()
            return captured

        model_summaries[model_name] = {"gpu": torch.cuda.get_device_name(0), "probe": {}}
        for prompt_length in PROMPT_LENGTHS:
            prompt = reference[:prompt_length]
            n_tokens = len(reference) - prompt_length
            model_summaries[model_name]["probe"][str(prompt_length)] = activation_probe(prompt)
            for seed in SEEDS:
                for arm in ("native", "sae_steering"):
                    set_seed(seed)
                    hook_shapes = {}
                    handle = None
                    if arm == "sae_steering":
                        def steer(_m, _i, output):
                            is_tuple = isinstance(output, tuple)
                            x = output[0] if is_tuple else output
                            shape = tuple(int(v) for v in x.shape)
                            hook_shapes[str(shape)] = hook_shapes.get(str(shape), 0) + 1
                            f = sae.encode(x).clone()
                            before = f[..., FEATURE_ID].detach().float()
                            f[..., FEATURE_ID] += STEERING_STRENGTH
                            after = f[..., FEATURE_ID].detach().float()
                            hook_shapes["feature_before_mean"] = float(before.mean().item())
                            hook_shapes["feature_after_mean"] = float(after.mean().item())
                            y = sae.decode(f)
                            return (y, *output[1:]) if is_tuple else y
                        handle = layer.register_forward_hook(steer)
                    start = time.perf_counter()
                    try:
                        with torch.inference_mode():
                            out = model.generate(prompt_seqs=[prompt], n_tokens=n_tokens,
                                                 temperature=1.0, top_k=TOP_K,
                                                 cached_generation=True, verbose=0)
                        sequence = str(out.sequences[0])
                    finally:
                        if handle is not None:
                            handle.remove()
                    elapsed = time.perf_counter() - start
                    if sequence.startswith(prompt) and len(sequence) == len(reference):
                        sequence = sequence[prompt_length:]
                    if len(sequence) != n_tokens:
                        raise RuntimeError(f"{model_name}/{prompt_length}/{seed}/{arm}: output length {len(sequence)} != {n_tokens}")
                    runs.append({"model": model_name, "prompt_length_bp": prompt_length, "seed": seed,
                                 "arm": arm, "sequence": sequence, "seconds": round(elapsed, 3),
                                 "hook": hook_shapes if arm == "sae_steering" else None})
        del model, base
        torch.cuda.empty_cache()
    lookup = {(r["model"], r["prompt_length_bp"], r["seed"], r["arm"]): r for r in runs}
    for row in runs:
        seq, p = row["sequence"], row["prompt_length_bp"]
        row["metrics"] = {
            "generated_region": bio_scoring.score_sequence(seq, organism_coding=True),
            "vs_real_generated_region": bio_scoring.compare_sequences(
                seq, reference[p:], organism_coding=True),
        }
        common = seq[300-p:] if p < 300 else seq
        row["metrics"]["common_coordinates_300_to_end"] = bio_scoring.compare_sequences(
            common, reference[300:], organism_coding=True)
        if row["arm"] == "sae_steering":
            native = lookup[(row["model"], p, row["seed"], "native")]["sequence"]
            row["metrics"]["vs_paired_native"] = bio_scoring.compare_sequences(
                seq, native, organism_coding=True)
    return json.dumps({"gpu": torch.cuda.get_device_name(0), "runs": runs,
                       "model_probe": model_summaries})


@app.local_entrypoint()
def main():
    reference = "".join(line.strip() for line in PHIX174.read_text(encoding="ascii").splitlines()
                        if line.strip() and not line.startswith(">")) .upper()
    if set(reference) - set("ACGT"):
        raise RuntimeError("Reference contains non-ACGT bases")
    print(f"Submitting 24 generations, GPU hard timeout {GPU_TIMEOUT_SECONDS}s", flush=True)
    raw = json.loads(run_factorial.remote(reference))
    out = {
        "status": "completed", "created_on": date.today().isoformat(),
        "design": {"phage": "phiX174", "accession": "NC_001422.1", "models": MODELS,
                    "prompt_lengths_bp": PROMPT_LENGTHS, "seeds": SEEDS,
                    "arms": ["native", "sae_steering"], "temperature": 1.0, "top_k": TOP_K,
                    "feature_id": FEATURE_ID, "strength": STEERING_STRENGTH,
                    "feature_strength_not_tuned": True, "gpu_timeout_seconds": GPU_TIMEOUT_SECONDS,
                    "timeout_cost_guard_note": "L40S function hard timeout; account billing still depends on current Modal rates."},
        "reference_length_bp": len(reference), "model_probe": raw["model_probe"], "runs": raw["runs"]
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "output": str(OUT), "gpu": raw["gpu"],
                      "generations": len(raw["runs"]),
                      "seconds": sum(r["seconds"] for r in raw["runs"])}, indent=2))


if __name__ == "__main__":
    main()
