# kmer-arm

Does adding a SpecMER-style k-mer candidate-selection step (Walton et al. 2025,
Eq. 2, adapted to DNA) on top of plain Evo2/EvSpark sampling push a full-length
phiX174 generation closer to the real genome's structure than native sampling
alone? Self-contained spinoff of the phiX174 full-length experiment in
`evo_benchmarking_tests` (native-vs-speculative baseline already run there,
`fulllength_bench_L40S.json`) -- this repo adds the third, k-mer-guided arm,
using the exact same prompt/seeds/model/checkpoint so results are directly
comparable.

## Files

- `kmer_table.py` -- k-mer table (counts -> probabilities) + SpecMER score.
- `build_kmer_table.py` -- builds `dependencies/microviridae_k346.npz` from
  `dependencies/microviridae_refs.fasta`.
- `kmer_fulllength_benchmark.py` -- **the k-mer arm.** Modal/GPU script: grows
  the phiX174 prompt chunk by chunk, each round sampling `c` candidates via
  EvSpark and keeping the one the k-mer table scores highest. Also loads
  `fulllength_bench_L40S.json` to compare the kmer arm directly against the
  native arm (the thing it's trying to beat, not a ground truth).
- `fulllength_benchmark.py` -- the native-vs-speculative baseline this is
  compared against (unmodified from the parent repo).
- `speculative_benchmark.py` -- shared Modal app/image/GPU config; both
  benchmark scripts import from it. Must stay colocated.
- `bio_scoring.py`, `genome_scoring.py` -- CPU-only scoring (GC, coding
  density, codon/k-mer divergence, ORF closure, GC-skew, terminator scan) run
  on every generated sequence, imported lazily by `kmer_fulllength_benchmark.py`'s
  local entrypoint after each Modal call returns. `genome_scoring.py` carries
  explicit `reliability_caveats` on its output (added 2026-11-03) after
  `Metrics_REPORT.md`'s damage-ladder calibration (a teammate's study, not
  committed to this repo, kept locally for reference) confirmed gene-shape-only
  metrics like Pyrodigal coding_density are blind to codon-shuffling damage --
  see that report and `KMER_ARM_ANALYSIS.md` before reading
  `coding_density`/`n_genes` deltas as biological validity rather than
  gene-shape plausibility.
- `phix174.fasta` -- the organism being generated (NC_001422.1).
- `dependencies/microviridae_refs.fasta` -- 20 *other* Microviridae genomes
  (~105.7kb) the k-mer table is built from: close coliphage relatives (G4,
  alpha3, ID18, ID52, WA13, ID32, NC35, St-1, phiK) plus deliberately
  divergent relatives across other hosts/subfamilies (Bdellovibrio, Chlamydia,
  Spiroplasma, marine and gut-metagenomic Gokushovirinae) -- chosen for
  diversity of signal, not just to dilute sparsity. Deliberately never phiX174
  itself -- see `build_kmer_table.py`'s docstring for the full list and why.
- `fulllength_bench_L40S.json` -- the already-collected native/speculative
  baseline (3 seeds, L40S) from the parent repo, kept here as the comparison
  target.

## Setup

```
python -m venv .venv
.venv/Scripts/activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install numpy modal
git clone https://github.com/dhnihaoya/EvSpark     # needed only for the real Modal/GPU run
```

Mock-test the k-mer arm's plumbing (no GPU, $0):
```
python kmer_fulllength_benchmark.py --mock
```

Real run (GPU-billed on Modal -- see cost/timing caveats in
`kmer_fulllength_benchmark.py`'s docstring before running all 3 seeds):
```
BENCH_GPU=L40S GPU_PRICE_PER_HR=1.95 modal run kmer_fulllength_benchmark.py
```

## Status

Run end-to-end on Modal/GPU (L40S) for all 3 seeds (0, 1, 2), scored against
the native baseline and the real genome -- `kmer_fulllength_bench_L40S.json`.

## Results (2026-11-03, 3 seeds, phiX174 full-length, score_mode="paper")

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| coding_density -- native | 0.906 | 0.223 | 0.859 | 0.663 |
| coding_density -- kmer | 0.877 | 0.662 | **0.125** | **0.555** |
| coding_density -- real | 0.927 | 0.927 | 0.927 | 0.927 |
| gc -- native | 0.460 | 0.419 | 0.376 | 0.418 |
| gc -- kmer | 0.320 | 0.346 | 0.332 | 0.333 |
| gc -- real | 0.449 | 0.449 | 0.449 | 0.449 |
| gc_skew sign changes -- native | 0 | 10 | 1 | 3.7 |
| gc_skew sign changes -- kmer | 0 | 4 | 2 | 2.0 |
| gc_skew sign changes -- real | 4 | 4 | 4 | 4.0 |
| terminator hits -- native | 44 | 25 | 34 | 34.3 |
| terminator hits -- kmer | 23 | 57 | 20 | 33.3 |

**No consistent improvement from the k-mer arm over native.** It only helped
on the one seed where native had already collapsed (seed 1: coding_density
0.223 -> 0.662), but on seed 2 -- where native was healthy (0.859, close to
real) -- the k-mer arm collapsed instead (0.125, the worst result across all
6 runs, n_genes dropping to 3). Averaged over all 3 seeds, k-mer's
coding_density is lower than native's, not higher. It also shows a
**consistent GC deficit on every seed** (0.32-0.35 vs real's 0.449, worse than
native on every seed) -- the one clean, reproducible pattern, and it's a
regression. GC-skew sign-change count is the one metric with a mild,
consistent edge toward k-mer (mean 2.0 vs native's 3.7, against real's
constant 4), though n=3 is too small to lean on. ORF closure stayed 0/7 for
both arms on every seed, as expected at this generation distance from a 64bp
seed -- uninformative either way (see `reliability_caveats` on
`genome_scoring.score_genome`'s output).

**Likely mechanism (investigated, not a coding bug):** `score_mode="paper"`'s
k-mer selection formula is k=3-dominated (P_k ~ 1/4**k means k=3 contributes
~79% of a chunk's score, k=6 only ~1%) and frame-blind (the table is built
over every nucleotide offset, not codon-phase-aligned). Full investigation,
including the raw Pyrodigal gene-length evidence and the untried
`score_mode="relative"` follow-up, in `KMER_ARM_ANALYSIS.md`.

**Caveat on every number above:** per `Metrics_REPORT.md`'s damage-ladder
calibration (a teammate's study, kept locally, not committed to this repo),
Pyrodigal `coding_density`/`n_genes` are near coin-flip
(AUROC ~0.59) at detecting codon-shuffling damage -- a sequence can score
high here with a scrambled, non-functional protein inside. These are
gene-shape/compositional-plausibility metrics, not biological validity.
Protein-level checks (Pfam/PHROG/Phold/CheckV) are out of scope for this
repo (teammate is running that heavier eval separately).
