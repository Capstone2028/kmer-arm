# SAE arm pilot: experiment report

**Run date:** 12 November 2026  
**Target:** phiX174, accession NC_001422.1  
**Status:** Completed  
**Full machine-readable output:** `phix174_sae_arm_experiment_20261112.json`

## 1. Summary

This pilot tested whether steering Evo 2 with a calibration-selected SAE feature
could improve simple sequence-quality measures compared with native Evo 2 and
an external k-mer selection method. We generated the same 5,322-base continuation
of phiX174 with four methods and three random seeds per method.

In this one-phage pilot, the SAE arm was closer to the real continuation than
native Evo 2 on average for GC content, predicted coding density, predicted gene
count, k-mer distribution, and codon distribution. Its k-mer and codon
divergences were lower than native Evo 2's on all three seeds. K-mer selection
did not improve the average k-mer or codon divergence over native generation.
The lower-temperature control was unstable: two seeds collapsed into highly
repetitive sequences.

These results are **a promising implementation and measurement pilot**, not
evidence that SAE steering improves phages in general. The run tested one
genome, used only three seeds, and did not test protein function or biological
viability.

## 2. Question and scope

The practical question was:

> When Evo 2 generates the rest of a small phage genome from a short prefix, do
> native sampling, a lower-temperature control, k-mer selection, or SAE
> steering produce continuations whose measured sequence statistics are closer
> to the real continuation?

This is an early development experiment for the SAE arm. It does not reproduce
the earlier team's Experiment 1 split or its full metric suite. The earlier raw
data folder was unavailable, so the dataset and calibration/test split below
were rebuilt from a new public-data snapshot. The phiX174 pilot uses only
composition and gene-shape measures; it does not include protein-family,
structure, infectivity, or viability assays.

## 3. Stage-by-stage work

| Stage | Plan and decision | What was done | Result |
|---|---|---|---|
| 1. Rebuild usable reference data | The prior raw folder was missing. Build a new, traceable dataset rather than claim an exact repeat of the old split. | Retrieved a fresh NCBI Microviridae snapshot; filtered genomes and translated annotated CDS records. | 7,649 records retrieved; 7,592 passed the length/base screen; 7,147 unique exact sequences remained; 2,549 met the CDS screen. |
| 2. Reduce train/test leakage | Keep near-identical genomes from crossing calibration and test groups. | Clustered genomes at 90% nucleotide identity with 80% bidirectional coverage; clustered proteins at 30% amino-acid identity with the same coverage rule. Selected one family-positive genome per genome cluster. | 2,549 candidate genomes formed 1,975 genome clusters. The five provisional family clusters occurred together in 273 genomes across 263 genome clusters. A deterministic split selected 30 calibration and 60 held-out test genomes. |
| 3. Verify the SAE can be used | Confirm the published SAE checkpoint and the Evo 2 layer hook before biological use. | Checked the checkpoint hash and tensor shapes; tested hook pass-through and intervention; ran a real Evo 2 + SAE GPU smoke check. | The SAE loaded at layer 26. The unsteered hook preserved logits exactly; steering changed logits. Both prompt-prefill and cached one-token decode hooks were observed. |
| 4. Select candidate features | Choose features using calibration genomes only, before inspecting held-out activation results. | Extracted layer-26 SAE values for 150 annotated genes from 30 calibration genomes. Ranked candidate features by within-genome activation difference between one family and the other four. | Five family-associated candidates were selected. One feature, 32710, was selected for two different families, which is a warning that some features are not family-exclusive. |
| 5. Validate without changing selection | Check whether the calibration-locked features retain their association on held-out genomes. | Measured four unique selected features across 300 family genes from 60 test genomes. A partial replication gene was excluded from that family's primary comparison. | The major-capsid feature 16115 was higher on the target gene in 57/60 test genomes (95%). See the family-level table below. |
| 6. Compare generation arms | Use the same model, prompt, length, and seeds so the methods can be compared fairly. | Ran native, temperature-control, k-mer-selection, and SAE-steering arms on one L40S GPU, with seeds 0, 1, and 2. | All 12 generations completed. |
| 7. Score and review | Compare the generated region to the actual phiX174 continuation, while keeping unreliable measures clearly labeled. | Scored composition, k-mer/codon distributions, Pyrodigal gene shape, homopolymers, and whole-genome diagnostic signals. | SAE was closest on several descriptive measures in this pilot. Exact-coordinate gene closure was 0/7 in every arm and is not useful for ranking these outputs. |

### Why these choices

| Choice | Reason |
|---|---|
| Start with phiX174 | It is a small, complete phage genome already available in the repository, so a full-length pilot is manageable and can be compared with a known reference. |
| Use a new 30/60 split | The previous raw folder was missing. The new split is reproducible and groups close genome matches before dividing calibration and test data. It must not be described as the original split. |
| Use `evo2_7b_262k` | The inspected SAE checkpoint and integration were tested with this model. Keeping the model fixed isolates the guidance method more cleanly. |
| Steer one feature, 16115 | The pilot needed one simple intervention. This calibration-selected feature was associated with the major-capsid family in held-out genomes. The association is provisional, not a semantic label guaranteed by the SAE. |
| Use a fixed +0.25 addition | A small fixed intervention was chosen for an initial end-to-end check. No steering-strength sweep was run, so this is not claimed to be an optimized value. |
| Keep three controls | Native generation is the baseline; lower temperature checks whether simpler sampling changes results; k-mer selection tests an external sequence prior against SAE's internal activation guidance. |
| Use three seeds | Three repeats reveal whether an observation is consistent across random starts, while keeping this first GPU run within a modest time budget. Three is still too few for strong statistical conclusions. |

## 4. Data preparation and split

Because the earlier raw data directory was not present, the team dataset was
recreated from a new NCBI retrieval. This is a new dataset snapshot and a new
split, not an exact reconstruction of the earlier experiment.

The initial screen returned 7,649 records. After requiring A/C/G/T sequence and
a 4,000–6,500 bp length, 7,592 remained. Exact duplicate removal left 7,147
sequences. Requiring at least six annotated CDS of 150 nt or longer left 2,549
candidate genomes. The five most common protein clusters were matched
provisionally to DNA pilot, internal scaffolding, major capsid, replication
initiator, and nonstructural proteins by their common product labels.

Genome clustering at 90% identity grouped the candidates into 1,975 clusters.
Of the genomes carrying all five provisional families, 273 represented 263
genome clusters. One representative per cluster was used in a deterministic
30-genome calibration / 60-genome test split (seed `20261112`). This helps
reduce leakage from close relatives, but the five protein clusters are still
provisional matches to the team's F1–F5 labels.

## 5. SAE feature selection and held-out check

The SAE checkpoint is a tied BatchTopK autoencoder for Evo 2 layer 26. Its
activation width is 4,096 and its feature width is 32,768. Feature selection
used the 30 calibration genomes only. Within each genome, the mean activation
of each family gene was compared with the mean activation across the other
four family genes. The ranking prioritized the median paired difference,
consistency across genomes, then mean difference. This was an exploratory
ranking, not a significance test.

| Provisional family | Calibration-selected SAE feature | Calibration genomes where target activation was higher |
|---|---:|---:|
| DNA pilot protein | 3575 | 30/30 (100%) |
| Internal scaffolding protein | 32710 | 29/30 (96.7%) |
| Major capsid protein | 16115 | 26/30 (86.7%) |
| Replication initiator protein | 29844 | 30/30 (100%) |
| Nonstructural protein | 32710 | 26/30 (86.7%) |

Held-out validation kept those feature IDs locked. It compared each feature's
activation on its target family gene against the other complete family genes
in the same test genome. Confidence intervals below are percentile bootstrap
intervals over genomes (5,000 resamples); they are descriptive and were not
adjusted for testing multiple features.

| Family | Feature | Test genomes compared | Target higher | Mean paired activation difference | Bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| DNA pilot protein | 3575 | 60 | 60/60 (100%) | 0.739 | 0.707–0.771 |
| Internal scaffolding protein | 32710 | 60 | 57/60 (95%) | 0.576 | 0.484–0.674 |
| Major capsid protein | 16115 | 60 | 57/60 (95%) | 0.659 | 0.588–0.718 |
| Replication initiator protein | 29844 | 59 | 59/59 (100%) | 0.711 | 0.636–0.786 |
| Nonstructural protein | 32710 | 60 | 57/60 (95%) | 0.550 | 0.379–0.773 |

One partial replication-family gene was excluded from that row's primary test,
leaving 59 comparisons. The duplicate feature 32710 and its activity across
multiple families are reminders that these feature IDs should be treated as
activation patterns associated with annotations, not as one-to-one biological
labels.

## 6. Four-arm generation design

All arms used `evo2_7b_262k`, the same 64-base phiX174 prefix, three seeds
(`0`, `1`, `2`), and a 5,322-base continuation. The prompt is the first 64
bases of NC_001422.1; each output therefore completes a 5,386-base linear
sequence.

| Arm | Setting | What it tests |
|---|---|---|
| Native | Temperature 1.0, top-k 4 | Standard sampling baseline. |
| Temperature control | Temperature 0.7, top-k 4 | Whether a simpler sampling change alters sequence statistics. |
| K-mer selection | Three candidates per 512-base chunk; retain the candidate with the best reference k-mer score | External guidance from a k-mer frequency table built from the other Microviridae references. |
| SAE steering | Add 0.25 to feature 16115 at layer 26 during prompt prefill and cached generation | Internal guidance using the calibration-selected major-capsid-associated feature. |

The steering signal was selected on calibration genomes and checked on held-out
genomes before this generation pilot. The pilot's primary readouts are general
composition and gene-shape statistics, rather than the same per-gene activation
difference used to select the feature.

## 7. Compute and completion

The real-model integration smoke test used an L40S and a 180-second function
timeout; its recorded resource ceiling was about $0.13. Calibration feature
extraction and held-out feature validation each used an L40S with a 900-second
timeout; each had an estimated resource ceiling of about $0.60. The final
four-arm generation used one L40S worker with a 3,600-second timeout and an
estimated resource ceiling of about $2.40.

The measured generation times summed to about 47.6 minutes across all 12
outputs: 7.9 minutes native, 7.6 minutes temperature control, 24.5 minutes
k-mer selection, and 7.6 minutes SAE steering. The known GPU-stage estimates
sum to about $3.73, plus possible small setup overhead. These are estimates,
not the actual Modal invoice; the run output does not contain a billing
receipt. Check Modal usage for the exact credit change.

The k-mer arm took about three times as long as each direct-generation arm
because it generated three alternatives for each 512-base chunk and scored
them before continuing. All 12 sequences and their detailed metrics were
saved in the JSON output listed at the top of this report.

## 8. Results

### 8.1 Average across the three seeds

The reference row is the 5,322-base continuation from the real phiX174 genome.
GC and coding density are fractions. A lower JS divergence means the
generated 4-mer or frame-0 codon distribution is closer to the real
continuation.

| Arm | GC | Coding density | Pyrodigal genes | Complete Pyrodigal genes | 4-mer JS divergence | Codon JS divergence |
|---|---:|---:|---:|---:|---:|---:|
| Real continuation | 0.4485 | 0.9271 | 7.00 | 5.00 | 0.0000 | 0.0000 |
| Native | 0.3916 | 0.7195 | 5.33 | 4.00 | 0.0649 | 0.0600 |
| Temperature control | 0.2295 | 0.8840 | 2.33 | 1.00 | 0.4499 | 0.4141 |
| K-mer selection | 0.3618 | 0.7344 | 5.67 | 4.67 | 0.0856 | 0.0779 |
| SAE steering | 0.4693 | 0.9156 | 7.67 | 6.67 | 0.0517 | 0.0552 |

**In simple terms:** SAE steering looked closest on these average measures,
although it predicted more complete genes than the real continuation. Its mean
4-mer and codon divergence were lower than native's. K-mer selection did not
improve those two divergence measures in this pilot. The low-temperature
control had a misleadingly high mean coding density because two outputs
collapsed into a single long predicted ORF; its diversity and composition
measures show that this was not a good result.

### 8.2 Results by seed

This table shows the individual runs so that the averages do not hide
seed-to-seed variation. `Homopolymer fraction` is the share of bases inside
same-letter runs of four or more bases; a high value can indicate repetitive
collapse.

| Seed | Arm | GC | Coding density | Genes / complete | 4-mer JS | Codon JS | Homopolymer fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Native | 0.3936 | 0.9532 | 6 / 4 | 0.07301 | 0.08010 | 0.0851 |
| 1 | Native | 0.4061 | 0.3585 | 6 / 5 | 0.05197 | 0.03797 | 0.0663 |
| 2 | Native | 0.3752 | 0.8467 | 4 / 3 | 0.06986 | 0.06200 | 0.1325 |
| 0 | Temperature control | 0.3925 | 0.9401 | 5 / 3 | 0.16362 | 0.12430 | 0.0432 |
| 1 | Temperature control | 0.1576 | 0.7120 | 1 / 0 | 0.45604 | 0.38970 | 0.2029 |
| 2 | Temperature control | 0.1385 | 1.0000 | 1 / 0 | 0.73005 | 0.72844 | 0.6240 |
| 0 | K-mer selection | 0.3395 | 0.7978 | 6 / 5 | 0.09159 | 0.09779 | 0.1385 |
| 1 | K-mer selection | 0.3166 | 0.6967 | 4 / 3 | 0.11698 | 0.09866 | 0.0887 |
| 2 | K-mer selection | 0.4292 | 0.7086 | 7 / 6 | 0.04824 | 0.03727 | 0.1003 |
| 0 | SAE steering | 0.4803 | 0.9070 | 5 / 5 | 0.04310 | 0.07400 | 0.0733 |
| 1 | SAE steering | 0.4339 | 0.8715 | 8 / 6 | 0.04547 | 0.03243 | 0.0964 |
| 2 | SAE steering | 0.4936 | 0.9682 | 10 / 9 | 0.06658 | 0.05915 | 0.0769 |

The SAE arm had lower 4-mer divergence than native for all three paired seeds,
and lower codon divergence for all three. The k-mer arm was closer than native
on both divergence measures for seed 2, but farther for seeds 0 and 1. This
illustrates why the three-seed average is useful, while also showing that
three repeats are not enough to characterize all possible outcomes.

### 8.3 Additional sequence diagnostics

| Measure | Real continuation | Native mean | Temperature mean | K-mer mean | SAE mean |
|---|---:|---:|---:|---:|---:|
| 4-mer complexity (0–1; higher is more diverse) | 0.9729 | 0.9686 | 0.6719 | 0.9521 | 0.9807 |
| Minimum stop-codon rate across frames (%) | 2.594 | 3.740 | 2.012 | 4.604 | 3.515 |
| CpG observed/expected | 0.9844 | 1.0862 | 1.2224 | 1.1125 | 1.0778 |
| Fraction of possible 5-mers absent | 0.0586 | 0.0505 | 0.5003 | 0.0898 | 0.0319 |
| Homopolymer fraction | 0.0644 | 0.0946 | 0.2900 | 0.1092 | 0.0822 |

The low-temperature arm's complexity, missing 5-mers, and homopolymer values
all point to repetition in seeds 1 and 2. SAE's averages are close to the real
continuation for these simple diagnostics, but these measures do not test
whether its predicted proteins have the correct family or function.

### 8.4 Whole-genome diagnostics and reliability

The whole-genome scorer compared generated sequences against the known
phiX174 annotation. It scored seven genes that were fully outside the 64-base
prompt; it skipped prompt-overlapping or origin-wrapping genes. **Every arm
closed 0 of those 7 genes at their exact real coordinates.** The scorer
explicitly says this exact-coordinate check is not a calibrated gene finder
and that near-zero closure is expected after generating thousands of bases
from a short prefix. Do not use this result to conclude that all arms have
equally poor genes.

The terminator proxy found an average of 29.3 hits for native, 7.0 for the
temperature control, 28.7 for k-mer, and 49.0 for SAE; the real phiX174 genome
itself produced 40 hits. The proxy is uncalibrated and produces false
positives, so these counts are recorded for completeness but are not treated
as a ranking metric.

## 9. Interpretation

The clearest pilot finding is that the SAE intervention ran end to end on the
actual Evo 2 model and, for these three seeds, produced continuations with
closer k-mer and codon statistics than native sampling. Its mean GC and
Pyrodigal coding density were also nearer the real continuation. These are
useful signs that the steering implementation is worth evaluating further.

The result does **not** establish that SAE is generally better than k-mer
selection. There is one target phage, only three random seeds, and the
differences across methods vary by seed. The pilot also used a different,
recreated split from the earlier team report. It did not run the protein-level
checks (such as family or homology checks) needed to tell whether generated
genes resemble the intended phage proteins.

## 10. Limits

- **One target:** phiX174 is a development case, not a representative sample of
  phages or proteins.
- **Small repeat count:** three seeds are useful for a first check but do not
  support strong statistical claims.
- **New data split:** the original raw folder was unavailable. The NCBI
  snapshot, family mappings, clustering thresholds, and 30/60 split were newly
  created and are not asserted to reproduce the earlier experiment.
- **Provisional family labels:** the five common protein clusters were matched
  to family names by annotation labels. They are candidate F1–F5 matches, not
  confirmed recovery of the team's original assignments.
- **Simple metrics only:** GC, k-mer/codon statistics, and Prodigal-style gene
  shape cannot establish protein identity, protein function, infectivity, or
  viability. No sequence from this run is a validated biological construct.
- **Metric caveats:** exact-coordinate ORF closure and the terminator scan are
  not reliable ranking signals here. Pyrodigal is designed for prokaryotic
  gene finding and may not match phage annotations perfectly.
- **Cost reporting:** the report records conservative resource estimates, not
  an actual billing amount. Modal's usage page is needed for the precise credit
  charge.

## 11. Recommended next work

1. Review the full JSON and representative sequences from each arm to confirm
   there is no implementation or scoring issue hidden by averages.
2. Keep feature selection locked to calibration data. Do not tune feature ID or
   steering strength on the held-out set.
3. If the pilot checks out, expand to multiple held-out phages and more seeds
   using the same four arms and predeclared metrics.
4. Add an evaluation that measures protein-family recovery for genes wholly
   after the prompt; report partial-overlap genes separately.
5. Compare methods across phage families before making a general claim. Keep
   mitochondria as a later, separate phase.

## 12. Files supporting this report

- Full run settings, generated sequences, and metrics:
  `phix174_sae_arm_experiment_20261112.json`
- Calibration-selected features:
  `microviridae_calibration_sae_feature_ranking_20261112.json`
- Held-out feature validation:
  `microviridae_test_sae_candidate_validation_20261112.json`
- Dataset and split details:
  `microviridae_split_manifest_20261112.json`
- SAE integration check:
  `modal_evo2_sae_gpu_smoke_20261112.json`
- Reproducible generation script:
  `../scripts/run_phix174_sae_arm_experiment_modal.py`
