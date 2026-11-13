# SAE experiment results

Generated SAE-arm outputs belong in this folder. Shared source records and
processed reference inputs remain in the repository-level `data/` folder.

The local preflight and synthetic Modal smoke test passed. The smoke test used
CPU only and confirmed that similar DNA/protein inputs cluster while unrelated
inputs stay separate. The full clustering also ran on CPU only.

Current results:

- 2,549 screened genomes formed 1,975 groups at 90% nucleotide identity.
- 19,049 proteins formed 4,945 clusters at 30% amino-acid identity.
- Five prevalent annotated protein clusters were found in 273 genomes across
  263 genome-similarity groups.
- A reproducible new split selects 30 calibration and 60 test groups using
  seed `20261112`, one genome per group.

The five protein clusters are provisional matches to the team's F1-F5 families.
The old raw split is unavailable, so this is a new dataset and split.

The public Goodfire SAE checkpoint was also inspected on Modal CPU. Its
published SHA-256 matched, and its 4,096-wide decoder / 32,768-feature tensor
shapes were recorded in `goodfire_sae_checkpoint_inspection_20261112.json`.
This metadata check did not run Evo 2 or use a GPU.

A CPU-only synthetic SAE hook smoke check passed for prefill-shaped and
one-token activations. It confirmed exact intervention-off pass-through,
feature-intervention output changes, finite outputs, and tuple cache
preservation. It used the published tied BatchTopK equations and no Evo 2
weights; see `modal_sae_hook_smoke_20261112.json`.

A real Evo 2 + SAE smoke check passed on an NVIDIA L40S. On the official
`evo2_7b_262k` model, intervention-off logits matched exactly, a one-feature
intervention changed logits, and the hook observed both prefill `[1, 4, 4096]`
and cached one-token decode `[1, 1, 4096]`. It used only a four-base random
prompt, no biological data, and a 180-second timeout (estimated resource
ceiling about $0.13). See `modal_evo2_sae_gpu_smoke_20261112.json`.

This proves the checkpoint and hook work together on the actual model. The
calibration-only feature extraction and held-out check are complete. Feature
16115, provisionally associated with major capsid, was higher on the annotated
major-capsid gene than the other four core families in 57/60 held-out genomes.
See `microviridae_test_sae_candidate_validation_20261112.json`.

A matched phiX174 pilot then completed on Modal: native Evo 2, a temperature
control, k-mer selection, and SAE steering, each with three seeds. The full
metrics and sequences are in
`phix174_sae_arm_experiment_20261112.json`; an English summary is in
`phix174_sae_arm_experiment_summary_20261112.md`. In this one-phage pilot, SAE
steering had lower k-mer and codon divergence than native generation for all
three seeds. This is encouraging but too small to establish general benefit,
and these metrics do not show protein function or phage viability.

For the staged experiment design, decisions, feature validation, per-seed
tables, metric explanations, compute estimates, and limitations, see the
fuller report `SAE_arm_pilot_results_20261112.md`.
