"""Inspect the public Goodfire SAE checkpoint on CPU, without saving it to Git."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
REPORT = RESULTS / "goodfire_sae_checkpoint_inspection_20261112.json"
REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"
FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"
EXPECTED_SHA256 = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.7.1",
        "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
)
app = modal.App("kmer-arm-goodfire-sae-cpu-inspection")


@app.function(image=image, cpu=2, memory=8192, timeout=300)
def inspect_checkpoint() -> str:
    import torch
    from huggingface_hub import hf_hub_download

    checkpoint_path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        state = loaded["state_dict"]
        wrapper_key = "state_dict"
    elif isinstance(loaded, dict):
        state = loaded
        wrapper_key = None
    else:
        raise TypeError(f"Unexpected checkpoint root type: {type(loaded).__name__}")

    tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    non_tensor_keys = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    return json.dumps({
        "repository": REPO_ID,
        "filename": FILENAME,
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "sha256_matches_published": digest.hexdigest() == EXPECTED_SHA256,
        "torch_version": torch.__version__,
        "root_wrapper_key": wrapper_key,
        "tensor_count": len(tensors),
        "tensor_keys_and_shapes": tensors,
        "non_tensor_keys": non_tensor_keys,
        "status": "PASS" if digest.hexdigest() == EXPECTED_SHA256 and tensors else "CHECK",
    })


@app.local_entrypoint()
def main() -> None:
    result = json.loads(inspect_checkpoint.remote())
    result.update({
        "inspected_on": date.today().isoformat(),
        "gpu_used": False,
        "compute_limit_seconds": 300,
        "estimated_runtime_resource_cost_usd": 0.02,
        "estimate_note": "Runtime-resource estimate only; image setup may add a small amount.",
    })
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
