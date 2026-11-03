# k-mer arm: why it isn't beating native, investigated

2026-11-03. Full 3-seed phiX174 full-length run (`kmer_fulllength_bench_L40S.json`,
`score_mode="paper"`, the SpecMER paper's own default) compared against the
existing native/speculative baseline (`fulllength_bench_L40S.json`) and the
real genome. This doc is the deep-dive behind the one-paragraph summary in
`README.md`'s Results section -- same numbers, full investigation of *why*.

## 1. The comparison, in full

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| coding_density -- native | 0.906 | 0.223 | 0.859 | 0.663 |
| coding_density -- kmer | 0.877 | 0.662 | **0.125** | **0.555** |
| coding_density -- real | 0.927 | 0.927 | 0.927 | 0.927 |
| n_genes -- native | 4 | 4 | 12 | -- |
| n_genes -- kmer | 3 | 11 | 3 | -- |
| n_genes -- real | 7 | 7 | 7 | -- |
| gc -- native | 0.460 | 0.419 | 0.376 | 0.418 |
| gc -- kmer | 0.320 | 0.346 | 0.332 | 0.333 |
| gc -- real | 0.449 | 0.449 | 0.449 | 0.449 |
| gc_skew sign changes -- native | 0 | 10 | 1 | 3.7 |
| gc_skew sign changes -- kmer | 0 | 4 | 2 | 2.0 |
| gc_skew sign changes -- real | 4 | 4 | 4 | 4.0 |
| terminator hits -- native | 44 | 25 | 34 | 34.3 |
| terminator hits -- kmer | 23 | 57 | 20 | 33.3 |
| kmer_vs_native kmer_js_divergence | 0.103 | 0.066 | 0.056 | -- |
| kmer_vs_native codon_js_divergence | 0.105 | 0.042 | 0.041 | -- |
| ORF closure (strict, /7) -- both arms | 0/7 | 0/7 | 0/7 | -- |

**No consistent improvement from the k-mer arm over native.** It only helped
on the one seed where native had already collapsed (seed 1: coding_density
0.223 -> 0.662), but on seed 2 -- where native was healthy (0.859, close to
real) -- the k-mer arm collapsed instead (0.125, the worst result across all
6 runs, n_genes dropping to 3). Averaged over all 3 seeds, k-mer's
coding_density is lower than native's, not higher.

A **consistent GC deficit on every seed** (0.32-0.35 vs real's 0.449, worse
than native on every single seed) is the one clean, reproducible pattern --
and it's a regression, not an improvement. GC-skew sign-change count is the
one metric with a mild, consistent edge toward k-mer (mean 2.0 vs native's
3.7, against real's constant 4), though n=3 is too small to lean on.
`kmer_vs_native` divergence (k-mer/codon JS) stayed small on every seed
(0.04-0.11), consistent with every prior native-vs-speculative comparison in
this project -- the k-mer arm isn't producing wildly different base/codon
statistics overall, just a different coding-density/GC outcome. ORF closure
stayed 0/7 for both arms on every seed, as expected at this generation
distance from a 64bp seed (see `genome_scoring.py`'s `reliability_caveats`)
-- uninformative either way.

## 2. What the Pyrodigal gene calls actually look like

Pulled the raw Pyrodigal-called gene lengths for seed 1 (the one seed where
k-mer's coding_density beat native's) to check whether the higher count/
density reflected anything like real gene structure:

| | n_genes | gene lengths (bp) |
|---|---|---|
| native | 4 | 126, 162, 414, 486 |
| kmer | 11 | 219, 261, 264, 273, 294, 312, 330, 336, 366, 372, 495 |
| real | 7 | 117, 171, 459, 528, 987, 1284, 1404 |

kmer's higher count isn't from finding a few long, real-structure-like genes
-- its longest called gene is only 495bp, while the real genome has four
genes over 500bp (528, 987, 1284, 1404). What's happening: the k-mer arm
produced a sequence with many consecutive ~200-500bp stretches, each
individually free of in-frame stop codons, so Pyrodigal chains together 11
short "complete" ORFs back-to-back instead of calling a few long ones. Ruled
out a boundary artifact first: gene starts/ends don't cluster near the
`chunk_len=512` round boundaries (checked all 11 gene boundaries against the
12 chunk boundaries -- distances scattered ±50-250bp, not near 0), so this
isn't the generation process literally restarting a frame every chunk.

## 3. Root cause: `score_mode="paper"` is k=3-dominated and frame-blind

Traced the actual mechanism steering candidate selection
(`kmer_fulllength_benchmark.py`'s `kmer_score`, mirroring `kmer_table.py`'s
`KmerTable.score`). Each round, 3 full candidate continuations are generated
and whichever scores highest under `score_mode="paper"` (SpecMER paper's own
Eq. 2, implemented correctly -- checked against the formula, no transcription
error) is kept. That formula sums raw per-position k-mer probabilities
across k=3,4,6, divided by length. Checked the actual table's magnitudes:

| k | mean P_k | 1/4^k |
|---|---|---|
| 3 | 0.0156 | 0.0156 |
| 4 | 0.0039 | 0.0039 |
| 6 | 0.000244 | 0.000244 |

Because P_k ~ 1/4^k by construction, k=3 contributes ~64x more to the sum
than k=4, and ~256x more than k=6 (confirmed numerically: for a 512bp chunk,
k=3 contributes ~79% of the total score, k=4 ~20%, k=6 ~1%). So in practice,
"paper" mode candidate selection is almost entirely driven by raw
trinucleotide composition -- k=6, the only one of the three with enough
resolution to carry anything resembling codon/reading-frame structure,
barely moves the decision at all.

Separately, and compounding it: `kmer_indices`/`count_kmers` build the table
and score candidates over every nucleotide offset (sliding by 1, not
codon-phase-aligned), on both strands. So even k=6 isn't reading-frame-aware
-- it can't specifically reward "no stop codon in the one correct frame,"
only "stop-codon-like trinucleotides are somewhat rare at this position, in
whichever frame happens to be open." That's consistent with what we found: a
sequence with diffuse, frame-switching open stretches rather than one
sustained real reading frame -- not a bug, but a genuine blind spot of this
exact scoring formula.

**Bottom line:** this matches what's already in `SESSION_NOTES.md`
("SpecMER's 'paper' score mode is small-k-dominated... favors keeping low k
around") -- a known, documented property of the formula, not a transcription
error in our implementation. But its practical consequence -- that k=6 is
nearly inert in candidate selection, and none of the k values are
frame-aware -- plausibly is why the output reads as "many short open
stretches" rather than "a few long real genes," and plausibly also explains
the consistent GC deficit (the formula directly optimizes composition toward
the reference set's aggregate GC, which doesn't match phiX174's own).

## 4. Next step, not yet tried

If you want the selection to actually weigh k=6 (closer to codon-level
signal), `score_mode="relative"` -- already implemented in `kmer_table.py`
and `kmer_fulllength_benchmark.py`'s inline copy, just unused by default --
multiplies each P_k by 4^k so all k's contribute equally. This would let k=6
actually participate in candidate selection, worth a follow-up run before
concluding k-mer guidance doesn't help at all, since this run only tested the
paper-default mode. Cheap to try: no GPU-side code changes, just pass
`--score-mode relative` to the local entrypoint.

## 5. Standing caveat on every result above

Per `Metrics_REPORT.md`'s damage-ladder calibration (a teammate's study, kept
locally, not committed to this repo), Pyrodigal `coding_density`/`n_genes`
are near coin-flip (AUROC ~0.59) at detecting codon-shuffling damage -- a
sequence can score high here with a scrambled, non-functional protein
inside. Everything in this doc is gene-shape/compositional-plausibility
analysis, not biological validity. Protein-level checks (Pfam/PHROG/Phold/
CheckV) are out of scope for this repo (teammate is running that heavier
eval separately).
