# Build notes

Running log of design decisions and open questions between GPU runs. Results
themselves live in `README.md` (summary) and `KMER_ARM_ANALYSIS.md` (deep
dive on why the first k-mer-arm run didn't beat native).

## 2026-11-03 -- amino-acid-guided arm, planning (not yet built)

**Motivation.** The nucleotide k-mer arm's `score_mode="paper"` scoring is
k=3-dominated (P_k ~ 1/4^k, k=3 is ~79% of a chunk's score, k=6 ~1%) and
frame-blind (table built over every nucleotide offset, not codon-phase-
aligned) -- see `KMER_ARM_ANALYSIS.md` section 3. It can only nudge raw base
composition, not reward a sustained, coherent reading frame. An amino-acid-
level scoring signal would directly target that gap.

**Why DNA was used in the first place, for the record:** not a choice made
in this repo -- inherited from the earlier `Capstone/kmer-evo2` project's
experiment guide (external .docx, not in this repo), which specifies
"SpecMER, Walton et al. 2025, Eq. 2, adapted to DNA" from the start. Likely
reason: Evo2 natively tokenizes/generates nucleotides only, so scoring in
that same space is free; amino-acid scoring requires committing to a reading
frame the model has no opinion on.

**The frame problem, and why it's smaller than it first looked.** Translating
a candidate to amino acids requires picking a frame before you know where a
gene starts or stops -- normally a real modeling problem. But this
experiment is phiX174-specific, and the whole generation setup is already a
strict linear continuation of the real genome's own coordinate frame (prompt
= `genome[:64]`, generation continues from position 64 -- the same
assumption `genome_scoring.orf_closure_check` already relies on). So the
correct frame at any absolute position isn't a guess -- it's a lookup:

- `genome_scoring.PHIX174_GENES` already has real 0-based coordinates for
  all 11 genes (sourced from NCBI `GCF_000819615.1/genomic.gbff`).
- For a candidate chunk spanning real positions `[lo, hi)`, any gene `g`
  whose span overlaps `[lo, hi)` has an exactly known phase at that position:
  `(lo - g.gene_start) % 3`. No "fewest stop codons" heuristic needed (that
  was the originally proposed fallback, now superseded for this organism).

**Remaining open design question:** phiX174's genes genuinely overlap in
*different* frames at the same position (the reason the Arc Institute phage-
design paper, biorxiv 2025.09.12.675911, had to write a bespoke overlap-aware
gene caller -- see `SESSION_NOTES.md` 2026-10-25). At a position where two
real genes are simultaneously active in two different frames, there isn't
one "the" frame to score against -- there are two correct ones at once.
Not decided yet: score a candidate against both active frames and require
both to look plausible, sum/average the two scores, or something else. This
is the one piece that still needs a decision before building the scorer;
everything else in the pipeline (chunking, candidate generation, best-of-c
selection) is unaffected and reusable as-is.

### Concrete steps once that's decided

1. **Reference proteins (CPU, free).** All 20 reference genomes in
   `dependencies/microviridae_refs.fasta` are RefSeq accessions (`NC_*`), so
   each almost certainly has annotated CDS features already -- same
   situation as phiX174's own `genomic.gbff`. Pull real translations via
   `efetch -db nuccore -id <acc> -format fasta_cds_aa` per accession (or
   batched) rather than running a gene caller on the reference side too
   (would just reinject the same gene-caller blind spots we're trying to
   get away from).
2. **Amino-acid k-mer table (CPU, free).** Same `KmerTable` machinery, 20-
   letter alphabet instead of 4-letter. Corpus shrinks ~3x in symbol count
   (~105kb nt / 3 ~= 35k amino acids total across 20 genomes), so k has to
   come down: k=2 (400 bins, ~88 counts/bin) and k=3 (8000 bins, ~4.4
   counts/bin) are plausible; k=4 (160,000 bins) would likely repeat the
   sparsity problem that already killed k=8 on the nucleotide table (see
   `build_kmer_table.py`'s docstring). Realistically `ks=[2,3]` only.
3. **Scorer.** Translate each candidate in the gene-coordinate-derived
   frame(s) active over its span (per the lookup above, with the overlap
   question resolved), score against the new amino-acid table. Swaps in for
   `kmer_score()` only -- the round loop/candidate generation in
   `kmer_fulllength_benchmark.py` is unaffected.
4. **Cost.** Steps 1-2 are CPU-only/free. A 3-seed GPU re-run to test it
   costs about the same as the completed nucleotide-arm run (sub-$1 total,
   per that run's per-seed cost).

**Not started.** No new files written yet for this (no fetch script, no
`build_aa_kmer_table.py`, no scorer changes). Next session should start by
resolving the overlapping-frame question above, since the scorer can't be
written sensibly until that's settled.

## 2026-11-10/11 -- block_pipeline.py: target-agnostic build/score/generate

**What got built.** A new, additive file, `block_pipeline.py` --
`kmer_fulllength_benchmark.py` and `kmer_guided_fulllength_benchmark.py` are
untouched (real GPU runs already behind them). Implements the test plan's
("K-mer arm test plan.docx") block-level design as three primitives plus the
orthology step they depend on:

- `make_blocks(gene_coords)` -- maximal non-overlapping segments, spacer
  folded into the block that follows, computed from any target's own
  `{name: (start, end)}` dict rather than hand-specified. For phiX174's 8
  non-wrap genes this produces exactly 4 blocks: `(C,D,F,G,H)` 50-964,
  `(E,)` 964-2284, `(J,)` 2284-2922, `(K,)` 2922-3917.
- `cluster_orthologs(corpus)` -- reference-free ortholog clusters via
  reciprocal-best-hit connected components over amino-acid 3-mer cosine
  similarity. No target genome participates in forming clusters.
- `build_tables(genomes, target_excluded, target_accession)` ->
  `{cluster_id: AAKmerTable}`, backoff-smoothed per cluster toward a pooled
  table (`counts + mu * pooled_probs`, `mu=1500` placeholder pending the
  real leave-one-genome-out sweep).
- `assign_target_genes(gene_seqs, tables)` -- the ONE per-target step: maps
  a specific genome's own named genes onto the existing clusters by
  nearest centroid. Same function for phiX174 or any Target B family
  member.
- `BlockScorer.score_candidate(dna, block)` -- translate each active
  gene's frame, hard-gate on premature/missing terminal stop, then mean PMI
  (`log P_observed - log P_null`, null = product of unigram amino-acid
  frequencies) across k-mer windows and active frames.
- `generate_block(prefix, block, c, generate_fn)` -- loops `c` seeds
  through a caller-supplied generator; no stop-codon filtering (that's the
  scorer's gate). Documented as NOT yet a real batched forward pass --
  EvSpark's `generate_native`/`native_sample_reference` are hard-coded to
  batch=1 (`EvSpark/evspark/api.py`), so this is a drop-in point for when
  real batching lands, not a finished optimization.

**Why it was rewritten mid-session.** The first version matched every
reference CDS against phiX174's OWN translated genes to decide per-gene
groups -- i.e. it used the target to define the orthology groups. That's
exactly the "you used phiX174 to define the groups" objection the test
plan's Target B section exists to dispose of, and it meant Target A's build
step could not have generalized to Target B's leave-one-out family sweep
without a rewrite. Fixed by making `cluster_orthologs` reference-free
(RBH over the corpus alone) and pushing everything target-specific into a
single downstream lookup (`assign_target_genes`). `prepare_target_a()` is
now the only phiX174-specific code in the file -- pure glue over four
target-agnostic functions; Target B's equivalent wrapper calls the same
four functions with a different genome, gene dict, and held-out accession.

**Validated (CPU-only `--mock`, no GPU, $0):** all 8 phiX174 genes assign to
8 distinct clusters (sizes 3-10 members); every real-genome block scores
finite (never gated); a base-shuffled control gates to `-inf` on every
block; `generate_block` produces the right number/length of candidates.

**Caveat that matters before spending GPU money: cosine similarity is not
BLAST.** `cluster_orthologs` approximates "reciprocal BLAST or annotation
matching" (test plan Risks section calls that step "the fiddliest build
step") with a CPU-only, dependency-free proxy: cosine similarity of whole-
sequence normalized amino-acid 3-mer *composition* profiles, reciprocal-
best-hit, connected components. This is a bag-of-3-mers comparison with no
notion of alignment, position, gaps, or local homology -- it can only ever
be as good as "do these two proteins use similar short amino-acid
vocabulary overall", which is a materially weaker signal than a real
alignment score:
  - **False merges.** Two unrelated proteins that happen to share amino-
    acid composition (common for small, compositionally-biased proteins --
    phiX174's gene H/K are both short and happen to be Lys/Arg/Gly-rich)
    can out-score true but divergent orthologs, especially at the 0.2
    similarity floor chosen empirically on this one 168-CDS corpus (never
    tuned against a known-correct orthology answer, because there isn't
    one in this repo yet).
  - **False splits.** A true ortholog pair that has diverged enough in
    local sequence (while conserving structure/function) can fail the RBH
    test even though a real alignment would call it a clear homolog --
    composition washes out exactly the signal alignment-based methods are
    built to find. This is the likely explanation for the fragmented
    cluster counts observed on the real data (82-114 clusters out of 168
    CDS at various thresholds, rather than the ~8-11 true Microviridae
    core gene families) -- see the `_tune_clustering`-style exploration
    this session, not committed as code, that produced those numbers.
  - **No confidence signal.** A real BLAST e-value/bit-score gives a
    principled accept/reject threshold; cosine similarity has none here --
    0.2 was picked by eyeballing cluster-size tables on one corpus, not
    derived from anything statistical.
  - **Consequence for the headline result.** If Target B's per-gene tables
    are quietly built from composition-matched-but-not-truly-orthologous
    groups, the PMI score stops measuring "does this look like gene X's
    family" and starts partly measuring "does this look vaguely similar in
    amino-acid usage to a grab-bag of proteins" -- weakening exactly the
    specificity-over-naive-counting claim the per-gene design exists to
    support (test plan, Target B section and Risks: "naive counting").

  **Before the real GPU run:** this should be replaced by (or at minimum
  spot-checked against) real reciprocal BLASTP or an annotation-based
  match (shared Pfam/InterPro domain, gene product name) for at least the
  Target B family sweep, where the orthology step is load-bearing for the
  method claim -- not strictly required for Target A, which only needs
  the pipeline shape to be exercised correctly, not the biology to be
  right.

## 2026-11-13 -- cluster_orthologs: annotation-first + real BLAST, replacing cosine

**What changed.** `cluster_orthologs` (`block_pipeline.py`) no longer uses
3-mer-cosine-similarity RBH as its primary signal -- the caveat two
sections up ("cosine similarity is not BLAST") is fixed, not just flagged.
New algorithm, annotation first:

1. **Curated annotation map** (`dependencies/ortholog_annotation_map.json`,
   NOT inferred in code -- an auditable, hand-checked data file): NCBI
   `/protein` text bucketed into specific, unambiguous categories
   (`MAJOR_CAPSID`, `LYSIS`, etc.), plus a `/gene` symbol -> bucket map for
   genomes that only use locus-tag shorthand ("gpA") with no descriptive
   text. Every phrase in the file was checked against all 168 CDS records
   to confirm it never matches two different loci within the same genome
   -- two candidates failed that check and were deliberately left out:
   "minor capsid protein" (reused for 2 different genes within
   NC_002643.1/NC_001741.1/NC_040350.1 alike) and "DNA binding protein"
   (same text on NC_015785.1's p6 AND p10). Both fall through to BLAST
   instead of being trusted as orthology evidence.
2. **Real NCBI BLAST+** (downloaded to `dependencies/tools/`, win64
   archive, not conda -- bioconda ships no win-64 build; see this entry's
   install notes below) runs all-vs-all over the WHOLE corpus, not just the
   unannotated leftovers -- deliberately, so the cohesion diagnostic in
   (3) has annotated records to compare against. Reciprocal-best-hit
   connected components (same per-genome-best + union-find shape as the
   old cosine version) bridge unannotated records into an existing
   annotation cluster when their BLAST component contains exactly one;
   otherwise they become their own new cluster (tracked as "orphan").
3. **Built-in check, not just a passive flag:** `cluster_orthologs` returns
   `(clusters, OrthologyDiagnostics)`. Diagnostics carry per-record
   provenance (curated / symbol / blast / orphan), per-cluster BLAST
   cohesion (does real sequence similarity actually support the
   annotation grouping -- e.g. `MAJOR_CAPSID: 18/25 agree`, correctly
   surfacing that the letter-named and VP-named capsid genes it merges are
   real but divergent), every tier-a/tier-b disagreement (gene-symbol
   bucket wins, logged either way), unmapped (unreviewed) `/gene` symbols,
   and undersized clusters. `cluster_orthologs`'s signature/output shape
   (`Dict[str, List[int]]` clusters, second return value) is unchanged in
   kind, so `build_tables` barely moved; `build_tables` now just prints
   this diagnostic.
4. **A\* excluded, not merged.** phiX174-style A* (internal in-frame start
   within gene A -- a truncated copy of A's own C-terminal region, not an
   independent ortholog) is dropped from the corpus entirely before
   clustering (`_is_nested_internal_start`/`_exclude_nested_products`),
   so it can't double-count that region in A's k-mer table. Caught
   generically (gene symbol ending "*", or product text with a bare
   "<letter>*" token) rather than hard-coded to phiX174: 3 records dropped
   on the real corpus, including two (NC_007856.1, NC_012868.1) that NCBI
   itself mislabeled `gene=A` with product text "protein A*" -- a real
   annotation inconsistency, not a hypothetical one.
5. **`assign_target_genes` also moved**, one step further than originally
   scoped: it still maps a target genome's own genes onto the clusters,
   but now checks the curated map first (phiX174's own gene letters are
   already keys in it) before falling back to cosine-nearest-centroid.
   Required once clusters stopped being cosine-defined themselves --
   cosine-matching against annotation-defined clusters was no longer
   self-consistent, and silently assigned two different phiX174 genes (C
   and G) to the same cluster on the first mock-test run after the rewrite.
   Only phiX174's K (no curated bucket at all) still uses the cosine
   fallback, which is not guaranteed collision-free -- known, documented
   limitation, not asserted away.

**BLAST+ install.** No binary was available via conda (bioconda's `blast`
package has no win-64 build). Downloaded NCBI's official win64 archive
directly (`ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/`) and
extracted to `dependencies/tools/ncbi-blast-2.17.0+/` -- no installer, no
system PATH/registry change, called by full path
(`_find_blast_bin_dir`). Gitignored (`dependencies/tools/`); re-download if
missing.

**Validated (CPU-only `--mock`, no GPU, $0):** all 8 phiX174 genes assign
to 8 distinct, biologically-labeled clusters (`C`->`MATURATION`,
`F`->`MAJOR_CAPSID`, etc. -- matches this corpus's own annotation); real
reference sequences never gated out; a base-shuffled control gates to
`-inf` on every block; cohesion diagnostic prints real, informative
disagreement (not 100% everywhere, as it shouldn't be).

## 2026-11-13 (continued) -- PHIX174_GENES was mislabeled, caught before any GPU spend

**The bug.** `PHIX174_GENES`' eight coordinate spans were exactly right;
every NAME attached to a span was wrong -- a one-position rotation across
the whole dict (the span at `(50, 221)` is gene K, not C; `(1000, 2284)` is
F, not E; etc. -- full corrected table below). Found by reviewing the
annotation-first rewrite above, independently confirmed by translating
each span from the real genome and checking residue counts against
phiX174's well-known protein lengths (F=427, D=152, J=38, G=175, H=328,
C=86, E=91, K=56) -- every single one matched the CORRECTED label, not the
label the code had. `kmer_fulllength_benchmark.py`'s `PHIX174_GENES_NOWRAP`
and `genome_scoring.PHIX174_GENES` carry the identical mislabeling (same
spans, same wrong names, clearly the original source per this file's
2026-11-10/11 entry) -- NOT fixed as part of this change, since those files
already have real completed GPU runs behind them and silently changing
what "real baseline" means for prior results needs a deliberate decision,
not a drive-by edit. Flagging here: **any interpretation of prior
native/kmer-arm GPU run results that relies on per-gene labels (not just
"the genome as a whole") should be treated as suspect until this is
checked there too.**

```python
PHIX174_GENES: Dict[str, Tuple[int, int]] = {
    "K": (50, 221), "C": (132, 393), "D": (389, 848), "E": (567, 843),
    "J": (847, 964), "F": (1000, 2284), "G": (2394, 2922), "H": (2930, 3917),
}
```

**Why both mock-test assertions passed anyway.** The premature-stop gate
only checks for stop codons, which doesn't care what the span is named; the
frames are internally consistent because the spans themselves were always
real gene boundaries. `real > shuffled` holds under almost any protein
table -- a real protein beats a shuffled one whether or not it's scored
against the right table. Both existing checks were insensitive to the
mislabeling by construction, which is exactly why it survived the first
mock-test run of the rewrite above undetected.

**Two checks added to `_mock_test` that would have caught it, and will
catch it again:**
1. `EXPECTED_RESIDUE_LENGTHS` -- translates each `PHIX174_GENES` span and
   asserts its length against the real protein's known residue count.
   Pins the labelling permanently; two lines.
2. A stronger discrimination test than the cluster-count check: score each
   real gene's own amino-acid sequence against EVERY cluster's table, not
   just its assigned one, and assert the assigned table wins. Hard-asserts
   for the 7 curated-symbol genes (all pass); only warns for K (no curated
   bucket -- see below), which is a real, bounded finding about K
   specifically, not a pipeline defect.

**Knock-on fixes, same session:**
- **`assign_target_genes` now tries a real BLAST reciprocal-best-hit
  bridge** (blast the target gene against the whole corpus reconstructed
  from each table's retained `member_seqs`/`member_accessions`, same
  bridging rule `cluster_orthologs` uses for its own leftovers) before
  falling back to cosine-nearest-centroid. With the corrected labels, K is
  the gene with no curated bucket; the BLAST bridge correctly resolves it
  to its own small `SYMBOL:K` cluster (9 members) rather than colliding
  with H's, which plain cosine had been doing.
- **`_rbh_components` edge-case fix:** when a candidate `j` has no hit at
  all back into `i`'s genome, every candidate scored `-inf` and `max()`
  was returning the first one by iteration order, which could spuriously
  equal `i` and fabricate a reciprocal edge where no real hit exists. Now
  guarded on the winning score actually being finite.
- **`build_tables` now prints the full cluster size distribution** and
  each cluster's implied backoff weight (`n_letters/(n_letters+mu)`, how
  much of its probability mass is real counts vs pooled backoff), plus the
  median-size check against the test plan's actual decision-gate rule
  (>=15 median -> per-gene-with-backoff viable). On the current 20-genome
  reference corpus: median cluster size is 1 (dominated by singleton
  leftover/orphan clusters) -- well below the per-gene threshold. The 8
  real target-gene clusters range 9-27 members and are the ones that
  matter for Target A; the test plan's per-gene-vs-pooled decision should
  be read off those, not the full 43-cluster distribution, which includes
  a long tail of unannotated singletons.
- `MAJOR_CAPSID` is a deliberately **shared** cluster (one table, 25
  reference members): in `cluster_orthologs`, annotated records keep their
  `ann_key` unconditionally, and `_cohesion_diagnostic` only reads that
  partition, never writes to it -- so BLAST disagreement (18/25 agree) is
  reported, not acted on. This is the intended design (annotation always
  wins, never silently overridden), not a bug.

## 2026-11-13 (continued) -- genome_scoring.PHIX174_GENES had the same mislabeling, now fixed

Checked `genome_scoring.py` for the same bug after fixing
`block_pipeline.py`'s copy: confirmed identical, same one-position
rotation, same 8 non-wrap genes (verified the same way -- translate each
span, check residue length against phiX174's known protein lengths). The
three origin-wrapping genes it also carries (A, A*, B -- `block_pipeline.py`
never had these) were already correct: B translates to exactly 120 aa; A
to 513 aa starting with a sequence matching the real replication-protein-A
motif; A* is mechanically consistent with A (starts 172 codons -- a
multiple of 3 -- into A's own frame, ending at the same stop, matching A*'s
known biology as an in-frame internal start within A).

**Why this was safe to fix immediately, unlike `kmer_fulllength_benchmark.py`
proper:** `orf_closure_check`'s pass/fail logic (`ok_len`/`ok_start`/
`ok_stop`/`ok_no_premature_stop`) reads only `segments`, never `name` --
the name is a pure reporting label. Both `kmer_fulllength_benchmark.py` and
`kmer_guided_fulllength_benchmark.py` only ever print/consume the
*aggregate* `n_closed`/`n_scored` (closure_fraction), never this per-gene
breakdown. So every completed GPU run's reported closure_fraction
(`fulllength_bench_L40S.json`, `kmer_fulllength_bench_L40S*.json`) is
numerically unaffected -- confirmed by re-running `orf_closure_check` on
the real genome before/after: `11/11` both times, same spans evaluated,
only the labels attached to each result changed. Fixed the dict's `name`
fields (segments untouched) plus the matching stale `# C`/`# D`-style
comments next to `kmer_fulllength_benchmark.py`'s `PHIX174_GENES_NOWRAP`
(a flat tuple list -- those comments were never read by code, cosmetic
only, but now correct).

## 2026-11-13 (continued) -- mu sweep run for real, DEFAULT_MU changed 1500 -> 750

**`sweep_mu` added** (`block_pipeline.py`, `--mu-sweep` CLI flag): leave-
one-genome-out log-likelihood sweep over the backoff strength mu, per the
decision table's "Smoothing strength (mu)" row. For each of the 20
reference genomes in turn, holds it out, runs the real annotation+BLAST
`cluster_orthologs` on the other 19 ONCE (mu-independent -- only the
backoff step depends on mu, so this doesn't repeat the BLAST call per mu
value), then scores the held-out genome's own real CDS translations
(looked up by their OWN annotation, same curated map `cluster_orthologs`
uses) against every candidate mu's backed-off counts.

**First pass used the wrong objective.** Initially reused
`AAKmerTable.score_aa`'s "relative" mode (`P_k * 20**k`, equal weight per
k) as the sweep's scoring function -- that's a ratio built for *ranking
candidates during generation*, not a proper scoring rule for model
selection. Sweeping mu against it came back monotonically favoring LESS
smoothing all the way down to the sweep's floor (mu=250, the smallest
value tried) with no interior maximum at all -- exactly backwards from
what a real log-likelihood should show (too little smoothing should get
punished when a held-out gene's rarer k-mers land on near-zero backed
probability). Replaced with `_log_likelihood_with_backoff`, real per-
k-mer-window log P under the backed-off model, properly weighted by
k-mer-window count (not residue count) when averaging across genes and
k's. That produces a genuine, well-behaved interior maximum.

**Result:** coarse sweep (250-8000) peaks at mu=750; a finer pass (500-1000,
step 50) confirms the peak sits at mu~700-750, extremely flat (5th-decimal
score differences between 700 and 800 -- the choice barely matters in
this range, which is itself useful to know). This is somewhat below the
test plan's prior expectation of 1000-2500 -- not a contradiction, just
the real swept answer superseding a guess. `DEFAULT_MU` changed from the
placeholder `1500.0` to `750.0`. Re-ran `--mock` after the change: still a
clean pass, backoff weights shifted as expected (e.g. `MATURATION`'s own-
weight moved from ~0.33 to ~0.50 -- less pooled backoff, as a lower mu
implies).

## 2026-11-13 (continued) -- prompts.fa confirmed as the real 300bp source, not yet actionable

> **CORRECTION (2026-11-14): the "first 300bp" claim below is FALSE.** It was
> confirmed as *sourced* (a real file, real genomes) but never checked
> *positionally*. The `__G2` records are 300bp windows at INTERNAL genome
> offsets of 68..3904; 0 of 25 start at position 0. See the 2026-11-14 entry
> at the end of this file -- this error invalidated an entire GPU run.

**What it is.** `prompts.fa` (repo root): 80 records, 40 distinct
Microviridae accessions x 2 variants each -- `__G1` is the first 1000bp of
that genome, `__G2` is the first 300bp. Confirms "~300bp" from an earlier
session question is real and sourced: it's the G2 variant of this file,
prepared by prepping the first 300bp of 40 real genomes as generation
prompts, not a number invented or guessed at. This is NOT the phiX174
prompt_len used by `kmer_fulllength_benchmark.py`/
`kmer_guided_fulllength_benchmark.py` (those stay at `prompt_len=64`,
untouched, same reasoning as every other entry in this file about not
touching files with real GPU runs behind them) -- `prompts.fa`'s 40
accessions are a disjoint genome set, confirmed zero overlap with the
20-genome reference corpus (`dependencies/microviridae_refs_cds_aa.fasta`)
`cluster_orthologs`/`build_tables` already use. That's good news for leak
risk (nothing in `prompts.fa` can accidentally end up inside the
reference corpus by name collision) but it also means this file is shaped
for the test plan's **Target B** (Microviridae leave-one-out family
sweep), not Target A (phiX174) -- a different, not-yet-built part of the
pipeline.

**The gap.** `prompts.fa` only has raw genomic DNA (for generation
prompts). It carries no CDS/protein annotation for these 40 genomes. The
test plan's actual Target B design ("each genome in the family is held
out in turn, tables are built from the rest") needs these 40 genomes'
own annotated CDS translations inside the reference corpus too -- not
just generation prompts, and not just the existing fixed 20-genome set.
That's a Stage 0 acquisition step (`efetch -db nuccore -id <acc> -format
fasta_cds_aa`, same approach `build_aa_kmer_table.py` already uses for the
current 20 genomes) that has NOT been run for these 40 accessions yet.
Deliberately not started this session -- recorded here as the next real
piece of Target-B work, not attempted speculatively.

## 2026-11-13 (continued) -- prompts.fa: isolate-origin check, coverage check, 3 removed

**All 40 are metagenome-derived, none cultured.** Checked every accession's
GenBank record (`/isolation_source`, `/host`, `/environmental_sample`,
REFERENCE title, assembly method) -- 37/40 matched an explicit metagenomic
signal (`MAG:` in 23, `ENV`/`environmental_sample` in 4, `UNVERIFIED` by
GenBank staff in 6, the rest named "microvirus" discoveries from faecal/
whole-organism virome surveys). The remaining 3 (MK249149, MW202414,
OP209174) read as ambiguous by keyword alone but are confirmed metagenomic
on closer read: MK249149 is a Kraberger/Varsani blackfly virome survey
(SPAdes assembly from whole-insect homogenate), MW202414 is annotated by
Cenote-Taker2 (a tool built specifically for virus contigs recovered from
metagenomes), OP209174 ("Bacteriophage N88_L5317_C7.5") has a contig-style
identifier and no named host. 40/40, no cultured isolates. Zero accession
overlap with either the existing 20-genome reference corpus or a separate
55-accession RefSeq Microviridae pull checked the same session (that pull
also turned out to be mostly metagenomic once sampled -- RefSeq membership
tracks curation, not cultured-vs-metagenomic origin).

**Gene-content coverage check** (`block_pipeline.py --coverage`, new
`target_coverage()`/`CoverageReport`): fetched `fasta_cds_aa` for all 40
(`dependencies/prompts40_cds_aa.fasta`). 9 have NO real CDS at all (7 of
the `UNVERIFIED` ones, including MG945467 whose genes are only
`misc_feature`/"similar to X" guesses with no `/translation`; plus 2
polar-lake MAGs, MT203029/MT204714, which also carry no translatable CDS
despite being labeled MAG) -- coverage is simply unmeasurable for these 9,
not zero; they were left in `prompts.fa` since generation prompts don't
need annotation, only the k-mer arm's per-gene scoring does. For the 31
genomes that do have real CDS, each gene was checked against the EXISTING
20-genome corpus's tables: does it resolve via curated annotation, via a
real BLAST reciprocal-best-hit bridge, or does it fall through to the
unreliable cosine-nearest-centroid last resort (same question
`assign_target_genes` asks, checked against the gene's OWN annotation
rather than assuming it shares phiX174's letter-naming convention).

Result: **67.3% trusted overall (96 curated + 38 blast / 199 genes)**, but
uneven -- most of the 31 land in the 60-100% range (close enough to the
existing corpus's gene families to be reliable), while 3 (OX614293.1 36%,
OX614603.1 29%, OX614619.1 43%, all rhesus-macaque-stool genomes annotated
by PHANOTATE with generic function labels) were clearly worse than
everything else, on both annotation match AND real BLAST similarity --
not just an annotation-quality artifact. **Those 3 accessions (6 records,
G1+G2) removed from `prompts.fa`** -- down to 37 genomes, 74 records.

**Per-genome reliability for the remaining 37** (curated/blast/fallthrough
out of total genes scored; `--` = no CDS annotation, coverage unmeasured):

| Tier | Accessions |
|---|---|
| High (>=80% trusted) | MH029519.1 (83%), MH029523.1 (83%), MH552523.2 (100%), MH616692.2 (100%), MH617265.1 (100%), MH617473.2 (100%), MK249149.1 (100%), MT135261.1 (100%, n=3), MT135264.1 (100%, n=1 -- weak sample), MW202414.1 (80%), MZ364261.1 (100%), PQ491838.1 (83%) |
| Medium (50-79%) | KT264759.1 (60%), MK496793.1 (50%), MK496799.1 (67%), MT135277.1 (50%, n=4), MT310155.1 (57%), MT310159.1 (62%), MT310332.1 (71%), MT310334.1 (67%), MW697720.1 (62%), OM869701.1 (71%), OP209174.1 (50%, n=2 -- weak sample), PQ006516.1 (71%), PQ006826.1 (71%), PQ007124.1 (62%), PQ007287.1 (71%), PQ007416.1 (71%) |
| No CDS annotation -- untested | MG945467.1, MG945474.1, MG945589.1, MG945594.1, MG945700.1, MG945809.1, MG945812.1 (all `UNVERIFIED`), MT203029.1, MT204714.1 (MAG, but no translatable CDS) |
| Removed (worst 3, see above) | ~~OX614293.1, OX614603.1, OX614619.1~~ |

Decision: run `prompts.fa` as-is (37 genomes) rather than expanding the
corpus first -- the median genome has enough coverage against the
existing 20-genome tables to be meaningful, and the corpus-expansion
question (RefSeq vs GenBank, cultured vs metagenomic) turned out not to
be a clean fix for the 3 that were actually bad, which were a specific
divergent lineage (PHANOTATE-called rhesus-macaque phages), not a general
annotation-quality problem solvable by pulling more genomes of any kind.
When reading k-mer-arm results per genome later, weight the medium-tier
and no-CDS genomes' per-gene endpoints accordingly -- their table coverage
is real but weaker, which is a property of the DATA, not the method.

**Correction (same day, later):** the native-only baseline for `prompts.fa`
is already done, by another part of the team -- this file's scope is
solely the k-mer arm itself. Superseded by the entry below.

## 2026-11-13 (continued) -- prepare_target_b: the generic per-target wrapper

**What was missing.** Everything up to this point (annotation-first
`cluster_orthologs`, `DEFAULT_MU=750`, `PHIX174_GENES` fix, `target_coverage`)
was reference-corpus-side work -- none of it needed touching to run a new
target. What WAS still missing: `prepare_target_a` is hardcoded to
phiX174 (its own gene letters, its own genome). There was no equivalent
for an arbitrary target genome. That gap is now filled by
`prepare_target_b(accession, genome, cds_fasta_path, tables, evalue)` --
same contract as `prepare_target_a` (takes the EXISTING tables, never
rebuilds them; returns `(gene_to_cluster, gene_coords, blocks, genome)`),
reusable for any target, not just the 28 genomes checked this session.

**Two real complications phiX174 never has, both handled the same way
phiX174's own A/A*/B (origin-wrapping genes) are already handled --
dropped rather than guessed at:**
- `join()`-location genes (found on 3 of the 28 checked genomes --
  2 real origin wraps, and one, PQ006826.1, a non-origin internal gap
  that isn't a wrap at all) are dropped individually, same policy as
  A/A*/B.
- A genome whose ENTIRE annotation is on the minus strand (a submitter's
  arbitrary choice of deposited orientation, not a biological anomaly --
  3 of the 28 checked genomes, including 2 of the 12 "high confidence"
  picks) is reverse-complemented as a whole and its coordinates remapped,
  which turns it into an ordinary forward-strand problem without
  touching `translate_genes`/`_gene_frame_aa`/`BlockScorer` at all. A
  genome with a genuine MIX of strands (not observed in these 28) would
  have its minus-strand genes dropped instead -- true mixed-strand
  handling isn't built, since nothing in hand needs it yet.

**Gene resolution reused, not reinvented.** `prepare_target_b` resolves
each gene via the SAME `_resolve_target_gene` helper `target_coverage`
already used (refactored out of it) -- checks the gene's own annotation
against the curated map, then a real BLAST bridge, falling back to
cosine-nearest-centroid only as the last resort (same three-tier chain
`assign_target_genes` uses for phiX174, applied here via the gene's own
product text instead of assuming it shares phiX174's letter-naming).

**Validated (`--validate-target-b`, CPU-only, $0):** fetched full genome
sequences for all 28 target genomes checked this session
(`dependencies/prompts28_genomes.fasta`) and ran every one through
`prepare_target_b` against the EXISTING 20-genome tables (unchanged).
**28/28: the real genome sequence survives BlockScorer's hard
premature-stop/terminal-stop gate on every block** -- the same
diagnostic-1-adjacent sanity check `_mock_test` runs for phiX174, now
passing on real external genomes, including the 3 reverse-complemented
ones and the 3 with a `join()` gene dropped. If the strand remap or frame
math were wrong anywhere, this is exactly the check that would have
caught it immediately (real biology should never hit a premature stop).
Several genomes show a gene-to-cluster collision (2+ of that genome's own
genes mapping to the same cluster) -- expected, not a bug: these are the
genomes `target_coverage` already flagged as having weaker per-gene
signal against the reference corpus, so some of their genes fall through
to the same weak cosine-nearest-centroid pick, same failure mode already
known from phiX174's own K gene.

**Not yet built:** the actual GPU-run driver that calls `prepare_target_b`
+ `generate_block` + a real Evo2 `generate_fn` for these targets. Everything
up to candidate generation is now in place and validated; wiring it to
the model is the next piece.

## 2026-11-13 (continued) -- block_kmer_benchmark.py: the Modal driver, and a real architectural gap found before any GPU spend

**Reused, not reinvented.** Surveyed every Modal-related file in both
`kmer-arm` and the parent `evo_benchmarking_tests` tree before writing
anything. Two independent Modal setups exist: `speculative_benchmark.py`'s
(app `evo2-speculative`, volume `evo2-hf-cache`, confirmed run -- real data
in `fulllength_bench_L40S.json`) and `Capstone/kmer-evo2/modal_app.py`'s
(more generic shape, but its own docstring says "UNTESTED until first
run"). Built on the proven one: `block_kmer_benchmark.py` imports
`GPU, PRICE, app, read_fasta_records, image, EVSPARK_DIR` from
`speculative_benchmark.py` exactly like `kmer_fulllength_benchmark.py`
already does, then `image.add_local_python_source(...)` to bake in
`block_pipeline.py`/`aa_kmer_table.py`/`kmer_table.py` as real importable
modules in the remote container -- NOT the older scripts' pattern of
hand-duplicating the scoring math inline (reasonable for a ~100-line k-mer
table, not for `cluster_orthologs`'s full annotation+BLAST pipeline).
Table-building/clustering itself still only ever runs locally, CPU-only,
before any `.remote()` call -- the remote function receives already-built
`AAKmerTable` objects (plain dataclasses of numpy arrays, picklable) as
plain arguments, same as any other Modal call.

**Hit `modal`'s own "Duplicate local entrypoint name: main"` error
immediately** -- `app` is the SAME shared `modal.App` object
`speculative_benchmark.py` already registers its own `main` entrypoint on;
named this script's entrypoint `block_kmer` instead (same reason
`kmer_fulllength_benchmark.py` is named `kmer_fulllength`, not `main` --
should have remembered that the first time).

**Real architectural gap found while wiring this up, before writing a
single line that would touch the GPU:** blocks from `make_blocks` almost
never land exactly on the 300bp prompt boundary (unlike phiX174's old
chunk-based script, which never used `make_blocks` at all). Added
`clip_blocks_to_prompt(blocks, gene_coords, prompt_len)` -- drops blocks
entirely inside the prompt, clips the one straddling block to start
exactly at `prompt_len`. First version only clipped the block's `start`
without re-checking which of its member GENES were still active past that
point; caught empirically on MH029523.1 at `prompt_len=300`
(`_gene_frame_aa`'s "stop codon falls outside this block's dna" -- a gene
inside that block had already ended before the clip point, same block,
different gene, still-unfiltered). Fixed by filtering `block.genes` to
only those whose own `gene_end > start` when clipping. Re-validated via
`block_pipeline.py --validate-target-b ... --prompt-len 300`: 28/28 real
target sequences pass the gate with real clipping applied, not just the
unclipped full-block check from earlier the same day.

**A second, deeper gap in the SAME area -- this one not just fixed, but a
real scope decision:** `prepare_target_b` reverse-complements the whole
genome for targets whose entire CDS annotation is minus-strand (3 of the
28: MH029519.1, MH029523.1, MT135261.1 -- all three "high confidence"
tier). That's fine for CPU-only scoring validation, but incompatible with
the actual GPU run: the external 300bp prompt is fixed in the ORIGINAL
deposited orientation (can't be flipped -- it's the input the native
baseline already ran against), while Evo2 only ever generates forward in
whatever orientation it's given. In the flipped (scoring) frame, the
block that comes "first" corresponds to the LAST part of the original
deposited sequence -- the part that only becomes available once nearly
the entire genome has already been generated. Block-by-block incremental
generation (generate one block, score c candidates, keep the best,
continue) cannot work for a whole-genome-flipped target; a "generate the
whole remaining sequence in one best-of-c shot, score the whole thing"
fallback would fix it, but wasn't built this session -- decided (not
assumed) to drop these 3 from this run rather than ship that fallback
un-validated. Run set is `prompts_reliable25_300bp.fa` (25 genomes), not
the 28-genome coverage-filtered set; `block_kmer_benchmark.py` also fails
loudly (not silently) if `prepare_target_b` ever reverse-complements a
genome it's asked to run, as a backstop against this happening again by
accident. Full reasoning in `prompts_description.txt` section 6.

**Validated end to end (`python block_kmer_benchmark.py --mock`, no GPU,
no Modal connection, $0):** all 25 target genomes x seed 0 run the
REAL round loop (table build -> `prepare_target_b` -> `clip_blocks_to_prompt`
-> `generate_block` -> `BlockScorer.score_candidate` -> argmax-pick ->
`bio_scoring` quality comparison against the real genome's own
continuation) with a fake random-codon sampler standing in for EvSpark --
zero exceptions, zero skips, 25/25 produced a quality readout. Not yet
run on an actual GPU.

## 2026-11-13 (continued) -- first real GPU run: CUDA OOM, lost whole batches, fixed

**Smoke test (1 genome, MT135264.1) passed on real GPU** -- 14.8 tok/s,
real quality comparison against the real genome worked. Confirmed the
whole pipeline (table build -> `prepare_target_b` -> generation -> scoring
-> quality) is correct end to end, not just in `--mock`.

**Then launched 3 parallel 8-genome batches (24 remaining genomes, split
7/7/10 then corrected to 8/8/8 per instruction) -- 2 of 3 crashed with
CUDA OOM, and ALL THREE batches lost every result, including the one
still running when it was killed as a precaution.** Root design flaw:
results were only written once, after a batch's ENTIRE loop finished --
a crash on genome 1 of 8 (what actually happened) meant nothing was ever
saved, and a crash on genome 5 of 8 would have erased genomes 1-4's
completed work too.

**The OOM itself, diagnosed in two stages:**
1. First theory: cross-call memory accumulation (inference_params from
   prior `generate_native()` calls not being freed). Added
   `torch.cuda.empty_cache()` after every call and re-ran the EXACT same
   genome (KT264759.1) that crashed before. Got the IDENTICAL numbers
   both times ("Tried to allocate 15.82 GiB", "26.27 GiB allocated by
   PyTorch") -- empty_cache() changed nothing, which rules out a leak.
2. Real cause: prefill processes the WHOLE growing context in one
   parallel FFT pass (vortex's `prefill_via_modal_fft`), so activation
   memory scales with context length and spikes hard at ~4-5kb -- a real
   peak, not a leak. The old chunk_len=512 script hits a similar peak at
   similar total context length but never in a single call large enough
   to need one contiguous 15+GB block the way this design's larger gene-
   sized blocks can. Fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_
   segments:True` on the image (must be set before CUDA initializes, so
   on the image's env, not inside the function) -- the fix PyTorch's own
   error message names for exactly this symptom (one large allocation
   failing despite enough nominal free memory -- fragmentation, not
   exhaustion). Not yet re-confirmed fixed on GPU (see below).

**The actual bug fixed -- losing whole batches, not just the OOM itself:**
- `block_kmer_bench_<GPU>[_<tag>].jsonl`: one line appended per
  (accession, seed) the MOMENT it finishes, from the LOCAL entrypoint
  (this machine's disk, not the remote GPU container's ephemeral
  filesystem -- a crash in the remote function can't erase what the
  local process already wrote for earlier genomes).
- Resumable: `_load_done_pairs` reads the existing JSONL on startup and
  skips any (accession, seed) already present -- a rerun after a crash
  costs only what's missing. A possibly-truncated last line (crash mid-
  write) is safely ignored, not treated as a resume failure -- tested in
  isolation, including the truncated-line case specifically.
- Per-genome `try/except` around the `.remote()` call: ANY failure (OOM
  or otherwise) is caught, logged to a separate `block_kmer_failures_
  <GPU>[_<tag>].jsonl`, and the run continues to the next genome rather
  than dying.
- `torch.cuda.max_memory_allocated()` logged per genome (`peak_gpu_mem_gb`
  in each result) -- visibility for next time, so a climbing-across-
  genomes number (real leak) is distinguishable from a flat, context-
  length-correlated one (the real prefill peak, what's actually observed).
- Genomes now processed longest-first (by real genome length) -- a GPU-
  headroom problem surfaces in the first call of a batch, not forty
  minutes in.

**Parallel-batch cost note (asked about, answered, then acted on):**
Modal bills GPU-seconds; running N genomes across N containers costs
about the same total compute as one container running sequentially, it
just compresses wall-clock. Real measured cost of the extra cold starts:
~60s/batch vs ~8-10s for a warm-reused call. The 3 crashed/killed batches
did burn real (small) money with zero results saved, which is exactly
what the checkpointing fix above is for.

## 2026-11-14 -- the block-kmer GPU run was invalid: wrong prompt frame, and a gate that rejected everything

Two independent bugs. Together they mean `block_kmer_bench_*.jsonl`
(23 genomes, L40S + A100) contains **no k-mer-guided generation at all**.
Do not report a guided-vs-plain contrast from those files.

**Bug 1: the prompts are not genome prefixes.** `prompts_description.txt`
section 1 and this file's own 2026-11-13 `prompts.fa` entry both state that
the `__G2` records are "the first 300bp" of each genome. They are not.
Verified by exact substring search of every prompt against
`dependencies/prompts28_genomes.fasta`: all 25 are found, on the + strand,
at offsets **68..3904**, and 18 of 25 land exactly on a CDS start. They are
gene-anchored windows. 0 of 25 start at offset 0.

`block_kmer_benchmark.py` trusted that claim and used `PROMPT_LEN = 300` as
an absolute *genome coordinate* in two places (`clip_blocks_to_prompt`, and
the `genome[300:300+n]` quality window). So for e.g. `KT264759.1` the model
was conditioned on real sequence from offset 3050 while `BlockScorer`
translated, gated and PMI-ranked its candidates in the frame of whichever
gene sits at 300..964. The guidance was applied to the wrong gene.

Consequences: block lengths (hence how much was generated) came from the
wrong frame, and every `quality.real` / `generated_vs_real` field in the
existing jsonl compares against the wrong genomic window.

**Bug 2: the hard gate rejected 100% of candidates, silently.**
`BlockScorer.score_candidate` returned a flat `-inf` unless a candidate
carried no stop codon across `stop_idx` residues AND placed one at exactly
that index. That is ~3e-12 by chance for a 553-residue gene --
unsatisfiable at block scale **regardless of bug 1**. Measured on the real
run: **98/98 rounds, all 23 genomes, every candidate `-inf`.**

The failure was invisible because `pick = max(range(len(scores)), key=...)`
over equal values returns index 0, and the caller appends unconditionally:

    scores = [-inf, -inf, -inf]  ->  pick = 0  ->  seq += cands[0]

So generation always produced complete, contiguous, plausible sequence
while no selection ever happened. `--c 3` tripled GPU cost for nothing.
This is exactly what `generate_block`'s own docstring warned against --
"rank an all-broken round by fewest violations rather than discard it
silently" -- a graded fallback specified but never implemented.

**What is NOT wrong.** Nothing was corrupted in transfer from Modal, and
there is no fabricated splice junction: `generate_block` passes the whole
current sequence as the prefix, so `prompt + block_kmer_seq` is one genuine
contiguous Evo 2 continuation. The sequences are a usable *unguided*
baseline; they are just not a k-mer arm.

### Fixes

- `block_pipeline.linearize_gene_coords(gene_coords, genome_len, prompt_end)`
  -- new. Re-expresses gene coordinates as signed offsets from the real
  `prompt_end`, so coordinate 0 *is* `prompt_end`. Genes behind the prompt
  on the circle come out ahead of it going forward; a gene straddling the
  boundary gets a negative start, the shape `_gene_frame_aa`'s
  `base = max(block_start, gene_start)` already handles. Sorting by that
  start gives true circular order, so `clip_blocks_to_prompt(blocks, lin, 0)`
  is the entire boundary clip. This also fixes circularity, which mattered
  independently: 21 of 23 generated spans cross the genome origin, and a
  purely linear "everything after prompt_end" view would drop every gene
  before it (for a prompt at 3050, most of the genome).
- `BlockScorer.score_candidate_detailed() -> (score, n_violations)` replaces
  the flat `-inf`. Violations = premature stops + missing terminal stop;
  score = `pmi_mean - 1000 * violations`. A clean candidate always outranks
  any violator, fewer violations outranks more, PMI breaks ties within a
  tier. PMI is computed per stop-free segment (`region.split("*")`) so a
  broken candidate still earns a real composition score instead of none.
  `score_candidate()` keeps the test plan's `-> float` signature.
- `block_kmer_benchmark._prompt_offset()` recovers the true offset by exact
  circular search and **skips** a genome whose prompt isn't found rather
  than guessing. `_circular_slice()` replaces `genome[300:300+n]`, which
  silently returned a short string for the 21/23 wrapping cases and so
  skipped the length-matched comparison without saying so.
- Loud guards for the failure mode that hid all this: per-round
  `violations` / `all_violating` / `degenerate_tie` in each record, a
  per-genome warning when every round is a tie, and a run-level
  `ERROR: NO SELECTION OCCURRED IN THIS ENTIRE RUN` banner.
- `evaluate_block_kmer.py` -- new. Rebuilds scorable sequences from the old
  jsonl using recovered offsets and circular natural-sequence windows, and
  refuses to label them a k-mer arm. The existing sequences stay valid as an
  unguided baseline; the useful thing is that unguided-continuous vs
  unguided-chunked gives a noise floor a real arm has to clear.

**Validation (no GPU, $0).** `python block_kmer_benchmark.py --mock` passes
on all 25 genomes: offsets recovered, blocks now spanning 3800-4400bp of
circular continuation, and `picked` genuinely varying (`[2,0,2,2]`,
`[0,2,2,0,1,0,2]`, ...) where it was `0` in all 98 rounds before. The
mock's sampler emits only non-stop codons, so it never exercises the
penalty path; that is covered by a direct ordering test on crafted
candidates (clean > 1 violation > 2 violations, nothing `-inf`).

**Not revisited here:** `genome_scoring.PHIX174_GENES`' one-position name
rotation. That was already fixed in its own right (see the 2026-11-13
"genome_scoring.PHIX174_GENES had the same mislabeling, now fixed" entry);
`block_pipeline.py`'s inline comment at `PHIX174_GENES` still claims it is
unfixed and is now stale. Label-only either way -- `orf_closure_check`
reads `segments`, never `name`.

**Still open:** the GPU rerun has not been done -- every number from the
block-kmer arm is currently unreportable.
