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
- `eval_metrics/bio_scoring.py`, `eval_metrics/genome_scoring.py` -- CPU-only
  scoring (GC, coding density, codon/k-mer divergence, ORF closure, GC-skew,
  terminator scan) run on every generated sequence.
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

Written and mock-tested (plumbing only: chunking, candidate seeding, best-of-c
selection, output length). **Not yet run on Modal/GPU** -- no real EvSpark call
has been made from this script. See `kmer_fulllength_benchmark.py`'s docstring
for the cost/timing caveats (prefill-cost is unmeasured for this model, so the
per-seed runtime estimate has a wide range) before spending GPU time on it.
