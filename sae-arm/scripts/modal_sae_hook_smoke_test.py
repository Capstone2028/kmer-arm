"""CPU-only synthetic check for the published Evo 2 SAE hook path.

This deliberately uses a small fake block with the same tensor and tuple
contract as the official Evo 2 SAE notebook. It does not load Evo 2 weights or
use biological sequences.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from math import prod
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
REPORT = RESULTS / "modal_sae_hook_smoke_20261112.json"
REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"
FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"
EXPECTED_SHA256 = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"
D_IN = 4096
D_SAE = 32768
K = 64

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
)
app = modal.App("kmer-arm-sae-hook-cpu-smoke")


@app.function(image=image, cpu=2, memory=8192, timeout=240)
def run_synthetic_hook_check() -> str:
    import torch
    from huggingface_hub import hf_hub_download

    torch.manual_seed(20261112)
    torch.set_grad_enabled(False)

    checkpoint_path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()

    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("_orig_mod.").removeprefix("module."): value
             for key, value in raw_state.items()}
    required = {"W", "b_enc", "b_dec"}
    if set(state) != required:
        raise AssertionError(f"Unexpected SAE keys: {sorted(state)}")
    if tuple(state["W"].shape) != (D_IN, D_SAE):
        raise AssertionError(f"Unexpected SAE matrix shape: {tuple(state['W'].shape)}")

    class TiedBatchTopKSAE(torch.nn.Module):
        """Inference equations from ArcInstitute's official SAE notebook."""

        def __init__(self) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(state["W"], requires_grad=False)
            self.b_enc = torch.nn.Parameter(state["b_enc"], requires_grad=False)
            self.b_dec = torch.nn.Parameter(state["b_dec"], requires_grad=False)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            pre = torch.relu(x @ self.W + self.b_enc)
            input_positions = prod(pre.shape[:-1])
            keep = K * input_positions
            top = torch.topk(pre.flatten(), keep)
            sparse = torch.zeros_like(pre.flatten()).scatter(0, top.indices, top.values)
            return sparse.reshape(pre.shape)

        def decode(self, features: torch.Tensor) -> torch.Tensor:
            return features @ self.W.T + self.b_dec

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

    class FakeEvo2Block(torch.nn.Module):
        def forward(self, x: torch.Tensor, cache: object) -> tuple[torch.Tensor, object]:
            return x, cache

    sae = TiedBatchTopKSAE().eval()
    block = FakeEvo2Block().eval()
    cases = []

    for case_name, sequence_length in (("prefill", 12), ("cached_one_token", 1)):
        activations = torch.randn(1, sequence_length, D_IN, dtype=torch.float32)
        cache = {"case": case_name}
        original_acts, original_cache = block(activations, cache)
        intervention_strength = {"value": 0.0}

        def intervention_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: tuple[torch.Tensor, object],
        ) -> tuple[torch.Tensor, object]:
            if not isinstance(output, tuple) or len(output) != 2:
                raise AssertionError("Expected an (activation, cache) block output")
            acts, state_cache = output
            if intervention_strength["value"] == 0.0:
                return output
            features = sae.encode(acts)
            if int(torch.count_nonzero(features)) > K * sequence_length:
                raise AssertionError("BatchTopK kept more than k activations per position on average")
            nonzero_per_feature = features.abs().sum(dim=(0, 1))
            feature_id = int(nonzero_per_feature.argmax())
            if nonzero_per_feature[feature_id] <= 0:
                raise AssertionError("Synthetic activations produced no active SAE feature")
            features = features.clone()
            features[..., feature_id] += 0.25
            return sae.decode(features), state_cache

        handle = block.register_forward_hook(intervention_hook)
        try:
            no_op_acts, no_op_cache = block(activations, cache)
            if not torch.equal(no_op_acts, original_acts) or no_op_cache is not original_cache:
                raise AssertionError(f"{case_name}: intervention-off hook changed model output")
            intervention_strength["value"] = 0.25
            steered_acts, steered_cache = block(activations, cache)
        finally:
            handle.remove()

        if steered_acts.shape != original_acts.shape:
            raise AssertionError(f"{case_name}: activation shape changed")
        if steered_cache is not original_cache:
            raise AssertionError(f"{case_name}: cache object was not preserved")
        if not torch.isfinite(steered_acts).all():
            raise AssertionError(f"{case_name}: SAE output contains non-finite values")
        if torch.equal(steered_acts, original_acts):
            raise AssertionError(f"{case_name}: feature intervention had no effect")

        cases.append({
            "case": case_name,
            "activation_shape": list(activations.shape),
            "output_shape": list(steered_acts.shape),
            "intervention_off_preserves_activations": True,
            "cache_preserved": True,
            "feature_intervention_changed_activations": True,
            "finite_output": True,
        })

    result = {
        "status": "PASS" if sha256 == EXPECTED_SHA256 else "CHECK",
        "checkpoint_sha256": sha256,
        "sha256_matches_published": sha256 == EXPECTED_SHA256,
        "tensor_shapes": {key: list(value.shape) for key, value in state.items()},
        "sae": "tied BatchTopK, k=64, d_in=4096, d_sae=32768",
        "batch_topk": "global top k * number_of_positions across sequence positions",
        "cases": cases,
        "gpu_used": False,
        "evo2_weights_loaded": False,
        "biological_sequences_used": False,
        "torch_version": torch.__version__,
    }
    if not result["sha256_matches_published"]:
        raise AssertionError("Checkpoint checksum did not match the published file")
    return json.dumps(result)


@app.local_entrypoint()
def main() -> None:
    result = json.loads(run_synthetic_hook_check.remote())
    result["run_date"] = date.today().isoformat()
    result["estimated_runtime_resource_cost_usd"] = 0.02
    result["estimate_note"] = "CPU runtime estimate only; image setup may add a small amount."
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
