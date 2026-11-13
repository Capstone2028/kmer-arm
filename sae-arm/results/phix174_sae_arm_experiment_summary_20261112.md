# phiX174 SAE arm pilot — 2026-11-12

## What was ran

One matched comparison ran on Modal using `evo2_7b_262k`, the first 64 bases of
phiX174 as the prompt, and three random seeds. Each arm generated the remaining
5,322 bases:

- Native Evo 2 (temperature 1.0, top-k 4)
- Temperature control (temperature 0.7, top-k 4)
- K-mer selection (three generated candidates per 512-base chunk, picked by the
  reference k-mer score)
- SAE steering (calibration-selected feature 16115, provisionally associated
  with major capsid, added at layer 26)

The selected feature had higher activation on the annotated major-capsid gene
than on the other four core families in 57 of 60 held-out genomes (95%). This
is evidence for an association in this dataset, not proof that the feature has
one exclusive biological meaning.

The run completed all 12 generations and the local scoring. Per-seed generation
times total about 47.6 minutes: roughly 7.9 minutes native, 7.6 minutes
temperature control, 24.5 minutes k-mer selection, and 7.6 minutes SAE
steering. The Modal function had a 60-minute timeout; the recorded cost ceiling
was an estimate up to about $2.40, not a billing receipt.

Full sequences, settings, metrics, and per-seed details are in
`phix174_sae_arm_experiment_20261112.json`.

## Mean metrics across three seeds

Reference values below are measured on the 5,322-base generated region of the
real phiX174 genome. JS divergence is lower when the generated sequence's
k-mer/codon distribution is closer to the real region.

| Arm | GC | Coding density | Predicted genes | Complete predicted genes | K-mer JS divergence | Codon JS divergence |
|---|---:|---:|---:|---:|---:|---:|
| Real phiX174 region | 0.4485 | 0.9271 | 7.00 | 5.00 | 0 | 0 |
| Native | 0.3916 | 0.7195 | 5.33 | 4.00 | 0.0649 | 0.0600 |
| Temperature control | 0.2295 | 0.8840 | 2.33 | 1.00 | 0.4499 | 0.4141 |
| K-mer selection | 0.3618 | 0.7344 | 5.67 | 4.67 | 0.0856 | 0.0779 |
| SAE steering | 0.4693 | 0.9156 | 7.67 | 6.67 | 0.0517 | 0.0552 |

Across these three seeds, the SAE arm had lower k-mer and codon divergence than
the native arm on every seed. Its mean GC, coding density, and predicted gene
count were also closer to the real-region values overall, although it
overpredicted complete genes on average. K-mer selection was farther from
native on k-mer/codon divergence and mean GC, but closer on coding density and
complete-gene count. The low-temperature arm was unstable: two seeds had one
very long predicted gene and high homopolymer fractions.

## Limits and interpretation

- This is a small development pilot on one phage and three seeds. It is an
  encouraging comparison, not a reliable estimate of performance across
  phages.
- The pilot evaluates composition and predicted gene shape. It does not test
  protein-family recovery, functional proteins, infectivity, or viability.
- Exact-coordinate ORF closure was 0/7 for every arm. The scorer explicitly
  warns that this is not a calibrated gene finder and that zero closure is
  expected at long range from this short prompt; do not use it to rank the
  arms.
- The SAE feature was selected using gene-family activation association on
  Microviridae calibration genomes. This pilot uses a held-out genome, but its
  small number of seeds and single target still limit the strength of the
  conclusion.
- These sequences are computational outputs and are not validated biological
  constructs.

## Next useful step

Treat this as a promising pilot signal. Before spending more GPU credits, review
the paired sequences/metrics, then run a larger held-out phage set with the same
four arms and predeclared evaluation measures. Keep any protein-family or
functional evaluation separate from the SAE feature-selection signal.
