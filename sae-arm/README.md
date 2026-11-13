# SAE experiment arm

This folder contains SAE-specific experiment planning, scripts, and generated
results. Shared public reference data and processed inputs remain in the
repository's `../data/` folder so the other experiment arms can reuse them.

## Progress and next step

The local preflight passed. The full dataset was clustered on CPU and a new
30-calibration/60-test split was created. See `results/README.md` and
`SAE_PHAGE_EXPERIMENT_PLAN.md`.

The SAE checkpoint metadata check and synthetic tuple-hook check passed on
Modal CPU. A real Evo 2 + SAE GPU smoke check then passed on an L40S, including
prefill and cached one-token generation. It used only a random four-base prompt
and was capped at 180 seconds (estimated resource ceiling about $0.13). See
`results/README.md` for what the check verifies and the next experiment stage.

The SAE checkpoint and real-model hook checks passed. Calibration-only feature
selection and held-out validation are complete. A three-seed phiX174 pilot
then compared native Evo 2, temperature control, k-mer selection, and SAE
steering on Modal. SAE steering had lower k-mer and codon divergence than
native generation on all three seeds in this single-phage pilot. See the full
staged report at `results/SAE_arm_pilot_results_20261112.md` and the complete
per-seed data at `results/phix174_sae_arm_experiment_20261112.json` before
choosing the next experiment.

To reproduce the local preparation steps from the repository root:

1. `.\.venv\Scripts\python.exe sae-arm\scripts\preflight_dataset.py`
2. `.\.venv\Scripts\python.exe sae-arm\scripts\export_candidate_proteins.py`
3. `.\.venv\Scripts\python.exe sae-arm\scripts\summarize_sequence_clusters.py`
4. `.\.venv\Scripts\python.exe sae-arm\scripts\make_experiment_split.py`

To rerun the Modal smoke test or CPU clustering, use
`.\.venv\Scripts\modal.exe run` followed by the relevant script path.
Generated SAE results belong in `results/`; shared inputs belong in `../data/`.
