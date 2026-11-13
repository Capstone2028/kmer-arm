# SAE checkpoint and integration gate

## Published checkpoint to inspect

- Repository: [`Goodfire/Evo-2-Layer-26-Mixed`](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed)
- File: `sae-layer26-mixed-expansion_8-k_64.pt`
- Published size: 537,020,630 bytes
- Published SHA-256: `598f131a2e71f1c49281335be1d11d2c91920881f91e2fd4e6aa0659e5b8b4b1`
- License: MIT
- Checkpoint type: BatchTopK, expansion 8, k=64, trained for Evo 2 layer 26 on mixed prokaryotic/eukaryotic data

The Evo 2 7B config has 32 blocks and hidden size 4,096. Arc's official SAE
notebook loads `evo2_7b_262k`, hooks block 26 (named `blocks-26` by its module
walker), and reads the activation from the first element of a tuple output.
It encodes `relu(x @ W + b_enc)`, keeps the global top `64 * number_of_positions`
values across the positions in the call, then decodes with `features @ W.T +
b_dec`. See the [official SAE notebook](https://github.com/ArcInstitute/evo2/blob/main/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb),
[Evo 2 7B config](https://github.com/ArcInstitute/evo2/blob/main/evo2/configs/evo2-7b-1m.yml),
and [Evo 2 hook implementation](https://github.com/ArcInstitute/evo2/blob/main/evo2/models.py).

## Checks required before generation

The checkpoint was inspected on Modal CPU on 2026-11-12. Its SHA-256 matches
the published value. The state dictionary has three tensors:

- `_orig_mod.W`: `4096 x 32768` float32
- `_orig_mod.b_enc`: `32768` float32
- `_orig_mod.b_dec`: `4096` float32

The inspection report is `results/goodfire_sae_checkpoint_inspection_20261112.json`.
The matrix shape and official inference equations establish the 4,096-to-32,768
orientation. A CPU-only synthetic hook check passed for prefill-shaped and
one-token activations, including intervention-off pass-through, feature
steering, and tuple cache preservation. See
`results/modal_sae_hook_smoke_20261112.json`. It used no Evo 2 weights.

The corresponding real Evo 2 GPU check passed on 2026-11-12 using an NVIDIA
L40S. It loaded `evo2_7b_262k`, confirmed intervention-off logits are exactly
unchanged, confirmed one SAE feature changes logits, and observed both prompt
prefill `[1, 4, 4096]` and cached one-token decode `[1, 1, 4096]` at block 26.
The prompt was a four-base random string; no biological data was used. The
report is `results/modal_evo2_sae_gpu_smoke_20261112.json`. The worker timeout
was 180 seconds, with an estimated resource ceiling near $0.13.

Run an intervention-off check where the hooked model produces the same logits
as the unhooked model. Then test one selected feature on a short synthetic
sequence and verify the hook works for both the initial prompt pass and token
generation. Only after those checks should the biological prompts be used.

The current EvSpark repository has native and speculative Evo 2 generation,
but no SAE steering implementation. The smoke script contains a prototype
adapter and has passed the synthetic model check; it still needs to be turned
into the experiment's reviewed generation code. Do not place this 537 MB
checkpoint in Git; download it into the Modal cache or a private persistent
volume when its run is approved.

## Biological pilot — 2026-11-12

Calibration and held-out SAE feature extraction completed. The matched
phiX174 four-arm experiment also completed on Modal: native Evo 2, temperature
control, k-mer selection, and SAE steering, three seeds each. Full data are in
`results/phix174_sae_arm_experiment_20261112.json`; the plain-English readout
is `results/phix174_sae_arm_experiment_summary_20261112.md`. The Modal
function's timeout was 60 minutes, and its estimated resource ceiling was
about $2.40. Check Modal billing for the actual charge.
