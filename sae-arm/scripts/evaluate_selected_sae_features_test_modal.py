"""Measure calibration-locked SAE features on held-out test genomes only."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
INPUT = RESULTS / "microviridae_test_sae_inputs_20261112.json"
CALIBRATION_SELECTION = RESULTS / "microviridae_calibration_sae_feature_ranking_20261112.json"
OUTPUT = RESULTS / "microviridae_test_sae_selected_features_20261112.npz"
MANIFEST = RESULTS / "microviridae_test_sae_selected_features_manifest_20261112.json"
REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"
FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"
EXPECTED_SHA256 = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"
TIMEOUT_SECONDS = 900

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3").run_commands(
    "python -m pip install evo2==0.6.0"
)
app = modal.App("kmer-arm-test-sae-feature-validation")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=TIMEOUT_SECONDS)
def extract_selected_features(input_payload: str, selected_feature_ids: list[int]) -> bytes:
    import numpy as np
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
    if digest.hexdigest() != EXPECTED_SHA256:
        raise AssertionError("SAE checkpoint checksum did not match the published file")

    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("_orig_mod.").removeprefix("module."): value
        for key, value in raw_state.items()
    }
    if set(state) != {"W", "b_enc", "b_dec"}:
        raise AssertionError(f"Unexpected SAE checkpoint keys: {sorted(state)}")

    class TiedBatchTopKSAE(torch.nn.Module):
        def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(state["W"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_enc = torch.nn.Parameter(state["b_enc"].to(device=device, dtype=dtype), requires_grad=False)
            self.b_dec = torch.nn.Parameter(state["b_dec"].to(device=device, dtype=dtype), requires_grad=False)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            values = torch.relu(x @ self.W + self.b_enc)
            positions = values.numel() // values.shape[-1]
            top = torch.topk(values.flatten(), 64 * positions)
            sparse = torch.zeros_like(values.flatten()).scatter(0, top.indices, top.values)
            return sparse.reshape(values.shape)

    payload = json.loads(input_payload)
    if payload.get("split") != "test" or payload.get("genome_count") != 60:
        raise AssertionError("Expected the 60-genome held-out test-only input bundle")
    if len(payload["records"]) != 60 or payload.get("family_gene_count") != 300:
        raise AssertionError("Test input counts do not match the 60 x 5 design")
    if len(selected_feature_ids) != 4 or len(set(selected_feature_ids)) != 4:
        raise AssertionError("Expected the four unique feature IDs locked on calibration")
    if min(selected_feature_ids) < 0 or max(selected_feature_ids) >= 32768:
        raise AssertionError("Calibration feature ID falls outside the SAE width")

    model = Evo2("evo2_7b_262k")
    base_model = model.model
    base_model.eval()
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    sae = TiedBatchTopKSAE(device=device, dtype=dtype).eval()
    selected_on_device = torch.tensor(selected_feature_ids, device=device, dtype=torch.long)

    modules: dict[str, torch.nn.Module] = {}

    def recurse(module: torch.nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = prefix + name
            modules[full_name] = child
            recurse(child, full_name + "-")

    recurse(base_model)
    if "blocks-26" not in modules:
        raise AssertionError("Could not locate Evo 2 block 26")
    target = modules["blocks-26"]

    values: list[np.ndarray] = []
    row_metadata: list[dict[str, object]] = []
    for record in payload["records"]:
        accession = record["accession"]
        token_ids = model.tokenizer.tokenize(record["sequence"])
        if len(token_ids) != int(record["sequence_length"]):
            raise AssertionError(f"{accession}: tokenizer positions do not match genome length")
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        captured: dict[str, np.ndarray] = {}

        def collect_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            activations = output[0] if isinstance(output, tuple) else output
            if not isinstance(activations, torch.Tensor):
                raise TypeError(f"Unexpected block output: {type(activations).__name__}")
            if activations.ndim != 3 or activations.shape[-1] != 4096:
                raise AssertionError(f"Unexpected block-26 shape: {tuple(activations.shape)}")
            features = sae.encode(activations)[0]
            selected = features.index_select(-1, selected_on_device)
            for region in record["family_regions"]:
                start, end = int(region["start_0"]), int(region["end_0"])
                if not (0 <= start < end <= selected.shape[0]):
                    raise AssertionError(f"{accession}: coordinates exceed token activations")
                captured[str(region["protein_id"])] = selected[start:end].float().mean(0).cpu().numpy()
            return output

        handle = target.register_forward_hook(collect_hook)
        try:
            with torch.inference_mode():
                base_model(input_ids)
        finally:
            handle.remove()
        if len(captured) != 5:
            raise AssertionError(f"{accession}: expected five feature summaries")

        for region in record["family_regions"]:
            protein_id = str(region["protein_id"])
            values.append(captured[protein_id])
            row_metadata.append({
                "accession": accession,
                "family_id": region["family_id"],
                "protein_id": protein_id,
                "product": region["product"],
                "start_0": region["start_0"],
                "end_0": region["end_0"],
                "strand": region["strand"],
                "partial_start": region["partial_start"],
                "partial_end": region["partial_end"],
            })

    matrix = np.stack(values).astype(np.float32, copy=False)
    if matrix.shape != (300, 4) or not np.isfinite(matrix).all():
        raise AssertionError(f"Unexpected held-out feature values: {matrix.shape}")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        selected_feature_values=matrix,
        selected_feature_ids=np.asarray(selected_feature_ids, dtype=np.int32),
        row_metadata_json=np.asarray(json.dumps(row_metadata)),
    )
    return buffer.getvalue()


@app.local_entrypoint()
def main() -> None:
    import numpy as np

    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    ranking = json.loads(CALIBRATION_SELECTION.read_text(encoding="utf-8"))
    selections = ranking["selected_calibration_candidates"]
    if len(selections) != 5:
        raise SystemExit("Calibration ranking must contain one locked feature per family")
    selected_ids = sorted({int(row["feature_id"]) for row in selections})
    result_bytes = extract_selected_features.remote(json.dumps(payload), selected_ids)
    with np.load(io.BytesIO(result_bytes), allow_pickle=False) as archive:
        values = archive["selected_feature_values"]
        returned_ids = archive["selected_feature_ids"].astype(int).tolist()
        row_metadata = json.loads(str(archive["row_metadata_json"]))
    if values.shape != (300, len(selected_ids)) or returned_ids != selected_ids:
        raise AssertionError("Held-out feature result did not match calibration-locked IDs")

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(result_bytes)
    manifest = {
        "status": "PASS",
        "created_on": date.today().isoformat(),
        "model": "evo2_7b_262k",
        "sae_repository": REPO_ID,
        "sae_sha256": EXPECTED_SHA256,
        "split": "test only",
        "candidate_selection_source": CALIBRATION_SELECTION.name,
        "selected_calibration_candidates": selections,
        "unique_feature_ids_evaluated": selected_ids,
        "genome_count": 60,
        "family_gene_count": len(row_metadata),
        "feature_value_matrix_shape": list(values.shape),
        "partial_family_gene_count": sum(
            bool(row["partial_start"] or row["partial_end"]) for row in row_metadata
        ),
        "gpu_used": True,
        "gpu": "NVIDIA L40S",
        "worker_timeout_seconds": TIMEOUT_SECONDS,
        "estimated_runtime_resource_cost_ceiling_usd": 0.60,
        "estimate_note": "900 seconds x L40S, 4 CPU cores, and 32 GiB memory at current published rates; setup may add a small amount.",
        "outputs": [OUTPUT.name],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
