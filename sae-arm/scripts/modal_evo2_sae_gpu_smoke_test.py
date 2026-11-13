"""Bounded Evo 2 + Goodfire SAE integration smoke check on Modal GPU.

Uses one deterministic random 4-mer only. No biological data or project FASTA
files are read. Model weights are downloaded into the temporary worker cache.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
REPORT = RESULTS / "modal_evo2_sae_gpu_smoke_20261112.json"
REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"
FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"
EXPECTED_SHA256 = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"
LAYER_PATH = "blocks-26"  # Hyphenated path used by Arc's official SAE notebook.
D_IN = 4096
D_SAE = 32768
K = 64
TIMEOUT_SECONDS = 180

# Official Evo 2 container supplies CUDA, PyTorch, and FlashAttention; adding
# Evo 2 itself follows ArcInstitute's documented container setup.
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3")
    .run_commands("python -m pip install evo2==0.6.0")
)
app = modal.App("kmer-arm-evo2-sae-gpu-smoke")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=TIMEOUT_SECONDS)
def run_integration_check() -> str:
    import torch
    from evo2 import Evo2
    from huggingface_hub import hf_hub_download

    torch.set_grad_enabled(False)
    torch.manual_seed(20261112)
    torch.cuda.manual_seed_all(20261112)

    checkpoint_path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_sha256 = digest.hexdigest()
    if checkpoint_sha256 != EXPECTED_SHA256:
        raise AssertionError("SAE checkpoint checksum did not match the published file")

    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("_orig_mod.").removeprefix("module."): value
        for key, value in raw_state.items()
    }
    if set(state) != {"W", "b_enc", "b_dec"}:
        raise AssertionError(f"Unexpected SAE checkpoint keys: {sorted(state)}")
    if tuple(state["W"].shape) != (D_IN, D_SAE):
        raise AssertionError(f"Unexpected SAE matrix shape: {tuple(state['W'].shape)}")

    class TiedBatchTopKSAE(torch.nn.Module):
        """Inference equations and flattened BatchTopK from Arc's SAE notebook."""

        def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(state["W"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_enc = torch.nn.Parameter(state["b_enc"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_dec = torch.nn.Parameter(state["b_dec"].to(device=device, dtype=dtype), requires_grad=False)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            values = torch.relu(x @ self.W + self.b_enc)
            positions = values.numel() // values.shape[-1]
            top = torch.topk(values.flatten(), K * positions)
            sparse = torch.zeros_like(values.flatten()).scatter(0, top.indices, top.values)
            return sparse.reshape(values.shape)

        def decode(self, features: torch.Tensor) -> torch.Tensor:
            return features @ self.W.T + self.b_dec

    def find_module(root: torch.nn.Module, target_name: str) -> torch.nn.Module:
        modules: dict[str, torch.nn.Module] = {}

        def recurse(module: torch.nn.Module, prefix: str = "") -> None:
            for name, child in module.named_children():
                full_name = prefix + name
                modules[full_name] = child
                recurse(child, full_name + "-")

        recurse(root)
        if target_name not in modules:
            sample = [name for name in modules if name.startswith("blocks-")][-8:]
            raise KeyError(f"Could not find {target_name}; nearby module paths: {sample}")
        return modules[target_name]

    model = Evo2("evo2_7b_262k")
    base_model = model.model
    base_model.eval()
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    sae = TiedBatchTopKSAE(device=device, dtype=dtype).eval()
    target = find_module(base_model, LAYER_PATH)

    sequence = "".join(random.Random(20261112).choices("ACGT", k=4))
    token_ids = model.tokenizer.tokenize(sequence)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    if len(token_ids) == 0:
        raise AssertionError("Synthetic sequence produced no model tokens")

    def logits_from(output: object) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unexpected Evo 2 forward output: {type(output).__name__}")
        return output

    with torch.inference_mode():
        baseline = logits_from(base_model(input_ids))

        hook_shapes: list[list[int]] = []
        mode = {"active": False}
        changed_feature_ids: list[int] = []

        def sae_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                if len(output) < 1 or not isinstance(output[0], torch.Tensor):
                    raise TypeError("Evo 2 layer output tuple did not start with an activation tensor")
                activations = output[0]
            else:
                activations = output
            if not isinstance(activations, torch.Tensor):
                raise TypeError(f"Unexpected block output: {type(activations).__name__}")
            if activations.shape[-1] != D_IN:
                raise AssertionError(f"Expected {D_IN} layer features; got {tuple(activations.shape)}")
            hook_shapes.append(list(activations.shape))
            if not mode["active"]:
                return output

            features = sae.encode(activations)
            active_strength = features.abs().sum(dim=tuple(range(features.ndim - 1)))
            feature_id = int(active_strength.argmax())
            if active_strength[feature_id] <= 0:
                raise AssertionError("SAE produced no active feature for the synthetic sequence")
            changed_feature_ids.append(feature_id)
            features = features.clone()
            features[..., feature_id] += 0.25
            reconstructed = sae.decode(features)
            if is_tuple:
                return (reconstructed, *output[1:])
            return reconstructed

        handle = target.register_forward_hook(sae_hook)
        try:
            off_logits = logits_from(base_model(input_ids))
            intervention_off_exact = torch.equal(baseline, off_logits)
            if not intervention_off_exact:
                raise AssertionError("Intervention-off hook changed Evo 2 logits")

            mode["active"] = True
            on_logits = logits_from(base_model(input_ids))
            intervention_on_changed_logits = not torch.equal(baseline, on_logits)
            if not intervention_on_changed_logits:
                raise AssertionError("SAE feature intervention did not change Evo 2 logits")
            prefill_shapes = list(hook_shapes)

            hook_shapes.clear()
            generation = model.generate(
                prompt_seqs=[sequence],
                n_tokens=2,
                temperature=1.0,
                top_k=4,
                cached_generation=True,
                verbose=0,
            )
            decode_shapes = list(hook_shapes)
        finally:
            handle.remove()

    if not prefill_shapes:
        raise AssertionError("The layer-26 hook did not run during prefill")
    if len(decode_shapes) < 2:
        raise AssertionError(f"Expected prefill plus cached decode hook calls; saw {decode_shapes}")
    if not any(shape[-2] == 1 for shape in decode_shapes):
        raise AssertionError(f"No one-token cached decode activation was observed: {decode_shapes}")

    result = {
        "status": "PASS",
        "model": "evo2_7b_262k",
        "sae_repository": REPO_ID,
        "sae_sha256": checkpoint_sha256,
        "layer_module_path": LAYER_PATH,
        "sae": "tied BatchTopK, k=64, d_in=4096, d_sae=32768",
        "synthetic_prompt": sequence,
        "intervention_off_logits_exact": intervention_off_exact,
        "intervention_on_changed_logits": intervention_on_changed_logits,
        "prefill_hook_shapes": prefill_shapes,
        "cached_generation_hook_shapes": decode_shapes,
        "steered_feature_ids": changed_feature_ids,
        "generated_sequence": str(generation.sequences[0]),
        "gpu_used": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "biological_data_used": False,
        "torch_version": torch.__version__,
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    return json.dumps(result)


@app.local_entrypoint()
def main() -> None:
    result = json.loads(run_integration_check.remote())
    result["run_date"] = date.today().isoformat()
    # 180s x L40S + 4 CPU cores + 32 GiB memory at current Modal rates.
    result["estimated_runtime_resource_cost_ceiling_usd"] = 0.13
    result["estimate_note"] = "Conservative resource ceiling from current published per-second rates; image setup may add a small amount."
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
