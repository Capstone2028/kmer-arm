"""Extract Goodfire SAE activations for calibration family genes on Modal."""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import date
from pathlib import Path

import modal


ROOT = Path.cwd()
RESULTS = ROOT / "sae-arm" / "results"
INPUT = RESULTS / "microviridae_calibration_sae_inputs_20261112.json"
OUTPUT = RESULTS / "microviridae_calibration_sae_gene_features_20261112.npz"
MANIFEST = RESULTS / "microviridae_calibration_sae_gene_features_manifest_20261112.json"
REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"
FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"
EXPECTED_SHA256 = "598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1"
TIMEOUT_SECONDS = 900

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.04-py3").run_commands(
    "python -m pip install evo2==0.6.0"
)
app = modal.App("kmer-arm-calibration-sae-feature-extraction")


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=TIMEOUT_SECONDS)
def extract_features(input_payload: str) -> bytes:
    import numpy as np
    import torch
    from evo2 import Evo2
    from huggingface_hub import hf_hub_download

    started = time.monotonic()
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
    if payload.get("split") != "calibration" or payload.get("genome_count") != 30:
        raise AssertionError("Expected the 30-genome calibration-only input bundle")
    if len(payload["records"]) != 30 or payload.get("family_gene_count") != 150:
        raise AssertionError("Calibration input counts do not match the expected 30 x 5 design")

    model = Evo2("evo2_7b_262k")
    base_model = model.model
    base_model.eval()
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    sae = TiedBatchTopKSAE(device=device, dtype=dtype).eval()

    module_paths: dict[str, torch.nn.Module] = {}

    def collect_modules(module: torch.nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = prefix + name
            module_paths[full_name] = child
            collect_modules(child, full_name + "-")

    collect_modules(base_model)
    if "blocks-26" not in module_paths:
        raise AssertionError("Could not locate Evo 2 block 26 using the official module path")
    target = module_paths["blocks-26"]

    feature_rows: list[np.ndarray] = []
    row_metadata: list[dict[str, object]] = []
    sequence_lengths: dict[str, int] = {}

    for record in payload["records"]:
        accession = record["accession"]
        token_ids = model.tokenizer.tokenize(record["sequence"])
        if len(token_ids) != int(record["sequence_length"]):
            raise AssertionError(
                f"{accession}: expected one nucleotide token per base; got "
                f"{len(token_ids)} tokens for {record['sequence_length']} bases"
            )
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        captured: dict[str, object] = {}

        def collect_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
        ) -> object:
            activations = output[0] if isinstance(output, tuple) else output
            if not isinstance(activations, torch.Tensor):
                raise TypeError(f"Unexpected block output: {type(activations).__name__}")
            if activations.ndim != 3 or activations.shape[0] != 1 or activations.shape[-1] != 4096:
                raise AssertionError(f"Unexpected block-26 activation shape: {tuple(activations.shape)}")
            if activations.shape[1] != int(record["sequence_length"]):
                raise AssertionError(f"{accession}: activation/token positions do not align")
            active = sae.encode(activations)[0]
            for region in record["family_regions"]:
                start = int(region["start_0"])
                end = int(region["end_0"])
                if not (0 <= start < end <= active.shape[0]):
                    raise AssertionError(f"{accession}: family coordinates exceed activation length")
                pooled = active[start:end].float().mean(dim=0).cpu().numpy()
                captured[str(region["protein_id"])] = pooled
            return output

        handle = target.register_forward_hook(collect_hook)
        try:
            with torch.inference_mode():
                base_model(input_ids)
        finally:
            handle.remove()

        if len(captured) != 5:
            raise AssertionError(f"{accession}: expected five pooled family-gene vectors, got {len(captured)}")
        sequence_lengths[accession] = len(token_ids)
        for region in record["family_regions"]:
            protein_id = str(region["protein_id"])
            feature_rows.append(captured[protein_id])
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
                "translation_table": region["translation_table"],
            })

    matrix = np.stack(feature_rows).astype(np.float32, copy=False)
    if matrix.shape != (150, 32768) or not np.isfinite(matrix).all():
        raise AssertionError(f"Unexpected feature matrix: {matrix.shape}, finite={np.isfinite(matrix).all()}")

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        gene_mean_features=matrix,
        row_metadata_json=np.asarray(json.dumps(row_metadata)),
        input_accessions=np.asarray([record["accession"] for record in payload["records"]]),
        sequence_lengths_json=np.asarray(json.dumps(sequence_lengths, sort_keys=True)),
    )
    return buffer.getvalue()


@app.local_entrypoint()
def main() -> None:
    import numpy as np

    input_payload = INPUT.read_text(encoding="utf-8")
    compressed_matrix = extract_features.remote(input_payload)
    with np.load(io.BytesIO(compressed_matrix), allow_pickle=False) as archive:
        matrix = archive["gene_mean_features"]
        row_metadata = json.loads(str(archive["row_metadata_json"]))
        accessions = archive["input_accessions"].tolist()
        sequence_lengths = json.loads(str(archive["sequence_lengths_json"]))
    if matrix.shape != (150, 32768) or len(row_metadata) != 150:
        raise AssertionError("Returned calibration activation matrix did not pass shape checks")

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(compressed_matrix)
    manifest = {
        "status": "PASS",
        "created_on": date.today().isoformat(),
        "model": "evo2_7b_262k",
        "sae_repository": REPO_ID,
        "sae_filename": FILENAME,
        "sae_sha256": EXPECTED_SHA256,
        "activation_layer": "blocks.26 / official module path blocks-26",
        "sae_architecture": "tied BatchTopK; k=64; global top 64 x sequence positions per forward call",
        "split": "calibration only",
        "genome_count": len(accessions),
        "family_gene_count": len(row_metadata),
        "families": sorted({str(row["family_id"]) for row in row_metadata}),
        "feature_matrix_shape": list(matrix.shape),
        "feature_matrix_dtype": str(matrix.dtype),
        "sequence_length_range_bp": [min(sequence_lengths.values()), max(sequence_lengths.values())],
        "all_selected_family_genes_complete": not any(
            row["partial_start"] or row["partial_end"] for row in row_metadata
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
