# Reliable25 300 bp SAE four-arm experiment

**Run status:** Completed  
**Prompts:** 25 phage sequences from the team's Reliable25 FASTA  
**Prompt length:** 300 bp  
**Generation model:** `evo2_7b` (1M context)  
**Outputs:** one unseeded output per prompt for each of four arms (100 total)

## 1. Summary

The full 25-prompt experiment completed and the saved file contains all 25 prompt indices, with all four methods present for each prompt. Each method generated a continuation targeted to the remaining length of its source genome.

The results do **not** show an overall win for SAE steering over native generation on the simple reference-similarity measures used here. Native generation had the lowest average 4-mer and codon divergence from the real continuation and the smallest average GC-content difference. SAE steering was usually the next closest on those measures. The k-mer selection arm was farther from the reference on average, and the lower-temperature control had the most variable results, including repeat-heavy sequences.

This is a useful end-to-end comparison across 25 phage prompts, but each prompt-method pair has only one unseeded sample. The results are descriptive and do not establish that any method is generally better across phages.

## 2. Experiment stages

| Stage | Plan and decision | Outcome |
|---|---|---|
| 1. Select inputs | Use the team's 300 bp Reliable25 prompt FASTA and its matched, oriented source-genome references. | 25 prompts passed the input manifest checks. |
| 2. Set the comparison | Keep the same prompt and target length for native sampling, lower-temperature sampling, k-mer selection, and SAE steering. Follow the team lead's instruction to generate without a fixed seed. | 100 sequences were produced: 25 per arm. |
| 3. Run on the 1M model | Use `evo2_7b`, matching the model choice used by the K-mer arm. | All prompts completed on Modal; the final checkpoint reports `completed`. |
| 4. Score outputs | Compare each generated continuation with the corresponding real continuation using sequence composition and gene-shape measurements. | Metrics are stored per prompt and arm in the JSON output. This review recomputed the summary metrics after removing whitespace from the two affected sequences. |
| 5. Review quality and limits | Check prompt coverage, arm coverage, output characters, output lengths, and method-level patterns. | Coverage is complete. Two raw sequences contain one space each; those are recorded below as data-quality issues. |

## 3. Main comparison

Values are **mean ± standard deviation across the 25 prompts**. For divergence and error measures, lower values mean the generated continuation's measured statistic was closer to the real continuation. The errors below are absolute differences, so direction does not cancel out across prompts.

| Arm | 4-mer Jensen–Shannon divergence | Codon divergence | Absolute GC difference | Absolute coding-density difference | Absolute gene-count difference | Absolute longest-gene difference |
|---|---:|---:|---:|---:|---:|---:|
| Native, temperature 1.0 | **0.0548 ± 0.0163** | **0.0617 ± 0.0217** | **3.4 ± 2.8 percentage points** | 12.9 ± 20.2 percentage points | **1.96 ± 1.86 genes** | **530 ± 443 bp** |
| Temperature control, 0.7 | 0.2352 ± 0.1916 | 0.2185 ± 0.1919 | 13.3 ± 10.6 percentage points | **7.9 ± 9.3 percentage points** | 3.72 ± 3.26 genes | 1,072 ± 1,207 bp |
| K-mer selection | 0.0922 ± 0.0381 | 0.0871 ± 0.0348 | 10.9 ± 5.0 percentage points | 15.4 ± 19.1 percentage points | 2.60 ± 2.06 genes | 601 ± 327 bp |
| SAE steering, feature 16115 at +0.25 | 0.0720 ± 0.0235 | 0.0698 ± 0.0236 | 4.8 ± 4.0 percentage points | 16.0 ± 15.3 percentage points | 1.96 ± 1.88 genes | 527 ± 310 bp |

### How often was an arm closer than native on the same prompt?

| Measure | SAE steering | K-mer selection | Temperature control |
|---|---:|---:|---:|
| 4-mer divergence | 7 / 25 | 3 / 25 | 1 / 25 |
| Codon divergence | 11 / 25 | 6 / 25 | 2 / 25 |
| GC difference | 9 / 25 | 1 / 25 | 4 / 25 |
| Coding-density difference | 10 / 25 | 10 / 25 | 13 / 25 |
| Gene-count difference | 9 / 25 | 8 / 25 | 5 / 25 |
| Longest-gene difference | 11 / 25 | 11 / 25 | 10 / 25 |

These are prompt-level counts, not statistical significance tests. The lower-temperature arm was closer on coding density in 13 prompts, but that measure alone misses its poor divergence and repeat-collapse behavior.

## 4. Results in plain English

- **Native generation was the strongest baseline overall.** Its average 4-mer and codon distributions were closest to those of the real continuations. It also had the smallest average GC difference.
- **SAE steering was competitive but did not beat native overall.** It was second on average for 4-mer and codon divergence and fairly close on GC. Its average coding-density difference was larger than native's, and it was closer than native on only 7 of 25 prompts for 4-mer divergence and 11 of 25 for codon divergence.
- **K-mer selection did not improve these reference-similarity measures in this run.** It was farther from native's baseline performance on average for divergence, GC, coding density, gene count, and longest-gene length. Its selection loop also took much longer.
- **The lower-temperature setting was unreliable.** Seven of its 25 outputs had 4-mer complexity below 0.70; one had a 3,557 bp homopolymer run. Its mean divergence was much higher and its variation much wider than the other arms.
- **The simple gene metrics are only rough checks.** Pyrodigal estimates coding regions and genes; these scores do not establish that the predicted proteins are the correct phage proteins or have a biological function.

## 5. Runtime and compute

Summed generation time recorded in the result file was approximately 54.8 minutes for native, 54.7 minutes for the temperature control, 55.5 minutes for SAE steering, and 178.5 minutes for k-mer selection. The k-mer method generated multiple candidates at each step, which accounts for most of its extra time. The total recorded arm time was about 5 hours 43 minutes; this excludes any separate billing components not captured by the per-arm timers.

The run metadata records a planning ceiling of **$20.80** for 13 estimated batches at $1.60 per batch. That is a budget estimate, not the actual Modal charge. The result file does not contain final account billing.

## 6. Data-quality checks and caveats

- The complete JSON includes prompt indices 0–24 exactly once and all four arms for every prompt. The recorded continuation lengths match the requested continuation lengths.
- Two raw sequence strings contain one space each: prompt index 16, native arm; and prompt index 19, temperature-control arm. Their stored character counts include the spaces, so each has one fewer actual A/C/G/T base than targeted. The summary above recomputes metrics after removing those spaces; the raw generated strings remain unchanged in the JSON for audit.
- The SAE feature, 16115, was selected and held-out validated with `evo2_7b_262k`, then transferred to `evo2_7b` for this run. The transfer was **not revalidated** on the 1M model. Its provisional “major capsid” association should not be read as a guaranteed biological meaning.
- There was one unseeded output for each prompt and method. This follows the team's instruction, but does not measure run-to-run variation for a prompt.
- GC, k-mer and codon statistics, Pyrodigal predictions, and sequence complexity are proxies. No protein-family recovery, structural analysis, infectivity, or viability test was performed.

## 7. Recommended next steps

1. Keep this result as a descriptive 25-prompt comparison; do not claim SAE superiority from it.
2. Before interpreting the two whitespace-containing outputs biologically, inspect the raw strings and decide whether to rerun those two prompt-arm cases or retain them as generation failures. The summary metrics here exclude the spaces but do not alter the raw outputs.
3. Validate the SAE feature's activation pattern on the 1M model before treating the feature association as model-independent.
4. If the team wants a stronger comparison, predefine additional samples per prompt or expand to a larger independent phage set, and add protein-level evaluation.

## 8. Files

- Full 25-prompt output, sequences, and per-prompt metrics: `reliable25_300bp_sae_four_arm_experiment_20261113.json`
- Preserved first-10-prompt handoff checkpoint: `reliable25_300bp_sae_four_arm_experiment_20261113_partial_10prompt_checkpoint.json`
- Generation and scoring orchestration: `../scripts/run_reliable25_sae_arm_experiment_modal.py`
- Prompt manifest and oriented references: `../inputs/reliable25_300bp_prompt_manifest.json` and `../inputs/reliable25_300bp_oriented_references.fna`
