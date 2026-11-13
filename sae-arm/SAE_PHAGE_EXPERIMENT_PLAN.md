# Phage SAE experiment plan

## Scope decision

Start with phiX174 as a development target. Move to a leave-one-genome-out
Microviridae comparison only after the data split, scoring, and steering code
are settled. Keep mitochondria as a later project phase.

This order follows the team's k-mer plan. phiX174 is a small, complete genome
with a baseline already in the repository, so it is the cheapest place to
find implementation problems. The phiX174 result is a development result;
the family-level comparison is the stronger method test.

## What Experiment 1 establishes

The 29 October design document describes a proposed calibration on at least
90 natural clusters, split 30/60, followed by a small Evo 2 check. It includes
15 candidate metrics, including ESMFold and Foldseek.

The later `# Experiment 1_results.md` report is the record of what was
actually done. It started with 7,581 NCBI genomes and found 229 usable
genomes in a broader Microviridae group defined by five shared protein
families (F1-F5). Only 17 clusters met the narrower phiX174-like definition.
The report uses 30 calibration and 60 test genomes from the broader group.
M14 ESMFold and M15 Foldseek were not run. The report's realized spend was
about $2.00-$2.30, below the design's initial $7-$22 estimate.

Use the completed calibration instead of repeating it. Describe its broader
Microviridae family clearly; do not call the 60 test genomes strictly
phiX174-like.

Use the report's metric findings as follows:

- Drop M7 ORF integrity. It scores some damaged genomes higher than natural
  genomes.
- Drop M3 closed-genome checks. It does not distinguish the tested damages.
- Treat M4 Pyrodigal and M5 PHANOTATE as gene-structure measures, not evidence
  that proteins are functional. They miss codon-shuffle damage.
- Treat M6, M11 Pfam, M12 PHROG, and M13 Phold as protein/family evidence with
  their stated limits. M6 is family-specific. Pfam and PHROG can label the
  wrong family readily. Phold was measured on only 20 genomes and is also
  generous to foreign proteins.
- Keep CheckV as a secondary completeness signal. Its whole-sequence score
  includes the prompt, so it cannot isolate generated-only coordinates.
- Report M10 separately. Synonymous recoding changes its RBS signal, so it is
  not a neutral DNA-motif control.
- Use M1 as a basic composition/length gate only.

No metric result is evidence of viability. The outcomes remain predictions
from sequence.

## Comparison design

Use the same target genomes, prompts, seeds, and decoding settings for four
arms: native Evo 2, a temperature control, k-mer selection, and SAE steering.
The current repository's phiX174 k-mer comparison has three seeds and only
composition/gene-shape metrics, so it is a development baseline. It is not a
protein-level four-arm result.

Choose steering features on the calibration genomes only. The primary
outcomes must be different from the signal used to select or steer those
features. A provisional design is to steer using a calibration-only subset of
the five shared gene families, then hold out the remaining core families and
protein-family metrics for evaluation. Finalize the held-out set after the
raw split and annotation tables are available.

For protein metrics, exclude genes copied from the prompt. Prefer genes that
start entirely after the prompt; report partial-overlap genes separately.
Keep CheckV as a whole-sequence diagnostic and show the prompt-only baseline
beside it.

## Data and compute gates

Local files already contain phiX174, 20 Microviridae references, their
recreated k-mer table, and GenBank records for those 21 accessions. The
GenBank inventory shows translation tables 1, 4, and 11 across the set. Read
the table for each genome when translating proteins or checking stops.

The Experiment 1 summary refers to additional files (`../data/split.tsv`,
`../data/genes_oriented.tsv`, metric tables, and a run log) that are not in this
repository. The prior raw folder is unavailable, so we must not claim an exact
replication of its split. A fresh NCBI snapshot was collected on 11 November
2026: 7,649 records, 7,592 passing the length/base screen, 7,147 distinct
exact sequences, and 2,549 candidates after requiring at least six translated
CDS features of 150 nt or longer and removing exact duplicates. See
`../data/README.md` and the accession/query manifest. These are candidate counts
only. A CPU-only Modal MMseqs2 job then grouped the 2,549 screened genomes at
90% nucleotide identity and their 19,049 proteins at 30% amino-acid identity,
using 80% two-sequence coverage. The genome clustering produced 1,975 groups.
The five most prevalent protein clusters are mostly annotated as major capsid,
replication initiator, nonstructural, internal scaffolding, and DNA pilot
proteins. 273 genomes carry all five clusters, spanning 263 of the 90%
genome-similarity groups. This is a provisional match to the team's five
families, not proof that we recovered the exact F1-F5 assignments.

A new deterministic split uses one family-positive genome from each similarity
group: 30 calibration groups and 60 test groups (seed 20261112). The split and
FASTA/protein inputs are in `results/`. This is a new split, not an exact
reconstruction of the missing prior raw split. The clustering thresholds are
new operational choices, not the team's original cluster definition.

Keep data checks and feature-label analysis on CPU where possible. Do not
launch a Modal GPU job until the exact run has a cost estimate and the account
has a hard spend limit within the available credit. Run GPU work in small,
logged stages and stop if a stage would exceed that limit.

## Next gates

Use calibration genomes only to choose SAE steering features. Keep primary
evaluation metrics separate from the steering signal. Before generation,
follow the checkpoint and integration gate in `SAE_CHECKPOINT.md`: the public
checkpoint hash/tensor keys and official BatchTopK equations are verified, the
synthetic CPU hook check passed, and the real `evo2_7b_262k` hook passed on
Modal L40S for prefill and cached one-token decode. Calibration-only feature
extraction and held-out validation are complete. The three-seed phiX174 pilot
in `results/phix174_sae_arm_experiment_20261112.json` compared native, lower
temperature, k-mer selection, and SAE steering. SAE steering was closer to the
real phiX174 region on several composition/gene-shape measures and had lower
k-mer/codon divergence than native on each of the three seeds. These are
promising pilot results, not a general conclusion: the test used one phage and
did not evaluate protein function or viability.

Next, review the result summary and compare full sequences and metrics. If the
pilot looks sound, plan a larger held-out phage comparison with predeclared
protein-family measures that do not reuse the feature-selection signal. Keep
the mitochondrial plan's plain-ReLU example separate; it is not the SAE
method used here.
