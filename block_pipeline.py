"""block_pipeline.py -- target-agnostic block-level build/score/generate.

See `K-mer arm test plan.docx`, "Shared design", "Target A -- phiX174" and
"Target B -- Microviridae leave-one-out". New and additive:
kmer_fulllength_benchmark.py and kmer_guided_fulllength_benchmark.py are
untouched (they already have real GPU runs behind them -- see
BUILD_NOTES.md 2026-11-10).

Everything in this file is written to be the SAME code for Target A
(phiX174, one development genome) and Target B (leave-one-out across the
Microviridae family) -- Target A is explicitly "scope validation" for the
method, not a separate pipeline, so the build step must not special-case
phiX174 anywhere. The only thing that legitimately differs between A and B
is which genome is "the target" for a given run (its own annotated gene
coordinates, used to cut blocks and to look up which table applies to
each gene) -- that's data, passed as an argument, not a branch in the code.

Four primitives:

    cluster_orthologs(corpus) -> {cluster_id: [seq_index, ...]}        # reference-free
    build_tables(genomes, target_excluded) -> {cluster_id: AAKmerTable}
    assign_target_genes(gene_seqs, tables) -> {gene_name: cluster_id}  # per-target lookup
    BlockScorer(tables, gene_to_cluster).score_candidate(dna, block) -> float
    generate_block(prefix, block, c, generate_fn) -> [dna]

Orthology, reference-free (fixes the earlier version of this file, which
built per-gene tables by matching every reference CDS against phiX174's OWN
translated genes -- exactly the "you used phiX174 to define the groups"
objection the test plan's Target B section calls out). NO target genome is
involved in forming clusters, same invariant as before; what changed
(2026-11-13) is how the reference corpus groups itself, replacing the
3-mer-cosine proxy with a real annotation-first + BLAST hybrid:

  1. Annotation first (cheaper and more accurate than any de novo
     clustering, for the ~57% of records that have it). Two tiers, highest
     priority first:
       a. Curated /protein text buckets (`_PRODUCT_BUCKETS`) -- specific,
          unambiguous phrases like "major capsid protein" or "DNA
          packaging protein" observed directly on this corpus
          (dependencies/microviridae_refs_cds_aa.fasta). Deliberately
          excludes generic catch-alls ("structural protein",
          "nonstructural protein") that curators reuse across genuinely
          different genes -- those fall through to BLAST instead of being
          trusted as orthology evidence.
       b. Raw /gene symbol (e.g. "F", "VP1", "p4"), with a data-driven
          unification pass (`_infer_gene_bucket_map`) first: if ANY record
          sharing a gene symbol across the corpus has product text that
          lands in a tier-a bucket, EVERY record with that gene symbol
          adopts that bucket -- even ones whose own /protein field is a
          bare locus-tag shorthand like "gpA" with no descriptive text.
          Without this, the same ortholog would split into two clusters
          purely because one genome's curator spelled the product out and
          another's didn't.
     Product-text buckets are deliberately subfamily-agnostic, so a
     phiX174-style gene (letter-named, e.g. "F") and a Gokushovirinae-style
     gene (VP-named, e.g. "VP1") merge into one cluster when curators used
     overlapping functional language for both ("capsid protein" /
     "viral coat protein VP1") -- this is real curator knowledge bridging
     sequence divergence that similarity-based clustering alone would miss
     (see BUILD_NOTES.md 2026-11-10/11 on why cosine-only clustering
     fragmented the family into 82-114 clusters instead of ~8-11).
  2. BLAST for everything else. One real blastp all-vs-all (NCBI BLAST+,
     downloaded to dependencies/tools/ -- see BUILD_NOTES.md), reciprocal-
     best-hit connected components over the WHOLE corpus (genome-scoped
     best hit, same per-genome-best + union-find logic as the old cosine
     version, just with blastp bitscore/E-value as the similarity source
     instead of a 3-mer cosine proxy). A record with no usable annotation
     (tier 1 found nothing) is assigned the annotation cluster of any
     annotated record sharing its BLAST component; if none of its BLAST
     component's members are annotated, it becomes its own new
     BLAST-only cluster. Annotation, where it exists, is never overridden
     by BLAST -- BLAST only places the leftovers.
  3. The built-in check this buys: `cluster_orthologs` also returns a
     cohesion diagnostic (see `_cohesion_diagnostic`) reporting, per
     annotation cluster, what fraction of its members actually land in the
     same BLAST component -- i.e. does real sequence similarity support
     the annotation grouping, printed by `_mock_test`. Known limitation
     surfaced by exactly this check on the real data: two records in
     NC_007856/NC_012868 are /gene-tagged "A" but their /protein text reads
     "protein A*" (the overlapping, differently-framed A* ORF, not gpA
     itself) -- a real NCBI annotation inconsistency this pipeline does not
     try to silently correct; it shows up as reduced cohesion on that
     cluster instead, for a human to look at.

Once clusters exist, a SPECIFIC target genome's own named genes (phiX174's
A-K, or any other family member's annotated genes for Target B) are
assigned into those existing clusters by nearest centroid -- the target
never participates in forming the clusters, only in being looked up
against them afterward. That lookup (`assign_target_genes`) is the only
per-target step in the whole pipeline; block cutting from the target's own
gene coordinates is the only other one. Both take the target's data as
plain arguments -- same functions run unchanged whether the target is
phiX174 (Target A) or genome #14 of 21 (Target B).

PMI scoring ("ranked by a mean log ratio of k-mer frequency over what
amino-acid composition alone predicts", shared-design section): for each
k-mer window, score = log(P_observed_kmer) - log(P_null_kmer), where
P_null_kmer is the product of each residue's marginal (unigram) amino-acid
frequency -- the "amino-acid composition alone" null. Composition is
measured over the whole reference corpus (not per small cluster, which
would be too noisy to trust at these group sizes) and shared across every
cluster's table.

Per-cluster tables are backoff-smoothed toward a pooled table built from
the same reference corpus (Katz-style additive backoff: cluster_counts +
mu * pooled_probs, renormalized -- decision table, "Smoothing strength
(mu)" row). Needed because cluster sizes off a 20-genome reference set are
small (median cluster size over the full 43-cluster set, including the
unannotated-leftover tail, is 1 -- `build_tables` prints the real
distribution every run), too sparse at k=3 (8000 bins) to trust on their
own. DEFAULT_MU is from the decision table's own prescribed check
(`sweep_mu`, a leave-one-genome-out log-likelihood sweep, 250-8000): the
real sweep (2026-11-13) found a genuine interior maximum at mu~700-750
(extremely flat -- 5th-decimal differences between 700 and 800), somewhat
below the plan's expected 1000-2500, not just the plan's prior guess.
"""
from __future__ import annotations

import gzip
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aa_kmer_table import AA, AAKmerTable, encode, kmer_indices, read_fasta, translate
from kmer_table import revcomp

_N_AA = len(AA)

# Real phiX174 protein lengths (residues, stop codon excluded), well-known
# and stable since the 1977/1978 original sequencing -- used by
# _mock_test to pin PHIX174_GENES' labelling: translating each span and
# checking its length against the NAME attached to that span is exactly
# the check that would have caught the 2026-11-13 mislabeling bug (every
# name shifted by one position) instead of passing silently.
EXPECTED_RESIDUE_LENGTHS: Dict[str, int] = {
    "C": 86, "D": 152, "F": 427, "G": 175, "H": 328, "E": 91, "J": 38, "K": 56,
}

# phiX174, 0-based half-open, non-origin-wrapping only -- same subset and
# source as kmer_fulllength_benchmark.PHIX174_GENES_NOWRAP /
# genome_scoring.PHIX174_GENES (A/A*/B excluded: they wrap the origin and
# don't exist in a linear one-shot continuation from genome[:prompt_len]).
# Target-A-specific; Target B supplies its own per-target gene dict of the
# same shape {name: (start, end)} -- nothing downstream of this constant
# cares which target it came from.
PHIX174_GENES: Dict[str, Tuple[int, int]] = {
    "K": (50, 221), "C": (132, 393), "D": (389, 848), "E": (567, 843),
    "J": (847, 964), "F": (1000, 2284), "G": (2394, 2922), "H": (2930, 3917),
}
# ^ Spans were always right; the NAMES zipped onto them were wrong (every
# single one, a one-position rotation) -- caught 2026-11-13 by translating
# each span against the real genome and checking residue counts against
# phiX174's well-known protein lengths (F=427, D=152, J=38, G=175, H=328,
# C=86, E=91, K=56 -- see _mock_test's EXPECTED_RESIDUE_LENGTHS assertion,
# added for exactly this reason). Inherited from genome_scoring.PHIX174_GENES
# (same mislabeling there too, NOT fixed as part of this change --
# kmer_fulllength_benchmark.py/kmer_guided_fulllength_benchmark.py already
# have real GPU runs behind them; flagged in BUILD_NOTES.md instead of
# silently touched).
PHIX174_ACCESSION = "NC_001422"  # never allowed into the reference corpus when target_excluded

DEFAULT_KS: Tuple[int, ...] = (2, 3)
DEFAULT_MU = 750.0  # leave-one-genome-out log-likelihood sweep, 2026-11-13 -- see sweep_mu, BUILD_NOTES.md
DEFAULT_EVALUE = 1e-5  # blastp RBH E-value cutoff -- see module docstring

# Scale separation for BlockScorer's graded violation penalty: a real PMI mean
# stays within single digits (it is a mean of log-ratios), so 1000 per
# violation guarantees any zero-violation candidate outranks any candidate
# with even one violation, while still ordering violators by how broken they
# are. Replaces the flat -inf hard gate -- see score_candidate_detailed.
VIOLATION_PENALTY = 1000.0


def _find_blast_bin_dir() -> Path:
    """Locate the local, non-system NCBI BLAST+ install (downloaded, not
    pip/conda-installed -- bioconda ships no win-64 build; see
    BUILD_NOTES.md). Picks the highest version if more than one is present."""
    candidates = sorted(Path(__file__).parent.glob("dependencies/tools/ncbi-blast-*/bin"))
    if not candidates:
        raise FileNotFoundError(
            "NCBI BLAST+ not found under dependencies/tools/ncbi-blast-*/bin -- "
            "download the win64 archive from "
            "ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ and extract it "
            "there (see BUILD_NOTES.md, 2026-11-13)."
        )
    return candidates[-1]


@dataclass(frozen=True)
class Block:
    genes: Tuple[str, ...]
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def make_blocks(gene_coords: Dict[str, Tuple[int, int]]) -> List[Block]:
    """Maximal non-overlapping segments; each intergenic spacer folded into
    the block that FOLLOWS it (not the one before), per the decision table.
    Target-agnostic: takes any target's own {name: (start, end)} dict. For
    PHIX174_GENES this yields exactly 4 blocks: (C,D,F,G,H) 50-964, (E,)
    964-2284 (the H-E gap folds into E), (J,) 2284-2922, (K,) 2922-3917."""
    ordered = sorted(gene_coords.items(), key=lambda kv: kv[1][0])
    blocks: List[Block] = []
    cur_genes: List[str] = []
    cur_start = cur_end = None
    for name, (s, e) in ordered:
        if cur_end is not None and s <= cur_end:
            cur_genes.append(name)
            cur_end = max(cur_end, e)
        elif cur_end is None:
            cur_genes, cur_start, cur_end = [name], s, e
        else:
            blocks.append(Block(tuple(cur_genes), cur_start, cur_end))
            cur_genes, cur_start, cur_end = [name], cur_end, e  # fold spacer into next block
    if cur_genes:
        blocks.append(Block(tuple(cur_genes), cur_start, cur_end))
    return blocks


def clip_blocks_to_prompt(
    blocks: Sequence[Block], gene_coords: Dict[str, Tuple[int, int]], prompt_len: int
) -> List[Block]:
    """For a GPU run whose prompt doesn't happen to end exactly on a block
    boundary (the common case for any real target other than phiX174's old
    chunk-based script, which never used make_blocks at all): blocks
    entirely inside the prompt have nothing left to generate, so they're
    dropped. The one block straddling the boundary (if any) is clipped to
    start exactly at prompt_len -- make_blocks guarantees block.start <=
    every member gene's own start, so a gene that began before prompt_len
    and continues past it is the mirror of a case BlockScorer already
    handles (a gene starting mid-block): _gene_frame_aa's
    `base = max(block_start, gene_start)` finds the first in-frame codon
    at/after prompt_len in THAT GENE's own real frame (established at its
    true start, not frame-0 at prompt_len), so translation of the
    genuinely-new tail is correct without needing to splice in any known
    prompt sequence.

    One thing this must NOT do: keep a gene in the clipped block's `genes`
    tuple if THAT gene's own end already falls at/before prompt_len --
    multiple genes can share a block (make_blocks groups overlapping
    genes together), and a short gene inside an otherwise-later-ending
    block can be fully consumed by the prompt while its block-mates are
    not. Scoring a fully-consumed gene against a clipped block raises
    (_gene_frame_aa's "stop codon falls outside this block's dna" --
    caught empirically on MH029523.1 at prompt_len=300 before this filter
    was added), since there's nothing left of it to translate. Blocks
    entirely after the prompt are returned unchanged."""
    out: List[Block] = []
    for block in blocks:
        if block.end <= prompt_len:
            continue
        start = max(block.start, prompt_len)
        if start == block.start:
            out.append(block)
            continue
        genes = tuple(g for g in block.genes if gene_coords[g][1] > start)
        out.append(Block(genes, start, block.end))
    return out


def linearize_gene_coords(
    gene_coords: Dict[str, Tuple[int, int]], genome_len: int, prompt_end: int
) -> Dict[str, Tuple[int, int]]:
    """Re-express a CIRCULAR genome's gene coordinates as signed offsets from
    `prompt_end`, so that forward-only generation visits genes in true
    circular order starting where the prompt actually stops.

    Why this exists (2026-11-14): block_kmer_benchmark.py used to clip blocks
    at the literal constant 300, on the documented but false assumption that
    every prompt was genome[0:300]. The real prompts are 300bp windows at
    internal offsets (68..3904), so blocks were cut and scored against
    whichever gene happened to sit near position 300 rather than the region
    the model was actually continuing. Passing a real `prompt_end` here is
    what fixes that; the linearization additionally handles the circularity
    that made the old framing wrong in a second way -- for 21 of 23 genomes
    in that run the generated span runs past the genome origin, so a purely
    linear "everything after prompt_end" view drops every gene before it.

    In the returned frame, coordinate 0 IS prompt_end, so:
      - a gene wholly ahead of the prompt keeps a small positive start;
      - a gene that lies "behind" prompt_end on the circle is ahead of it
        going forward, so it gets a start up to genome_len away;
      - a gene STRADDLING prompt_end gets a NEGATIVE start, the same shape
        clip_blocks_to_prompt/_gene_frame_aa already handle for a gene that
        begins inside the prompt (base = max(block_start, gene_start)).
    Sorting by this start therefore yields circular generation order, and
    `clip_blocks_to_prompt(blocks, lin_coords, 0)` does the boundary clip.
    """
    lin: Dict[str, Tuple[int, int]] = {}
    for name, (s, e) in gene_coords.items():
        span = e - s
        behind = (prompt_end - s) % genome_len  # how far back prompt_end is from this gene's start
        d = -behind if behind < span else (s - prompt_end) % genome_len
        lin[name] = (d, d + span)
    return lin


DEFAULT_MAX_BLOCK_BP = 600


def split_long_blocks(
    blocks: Sequence[Block],
    gene_coords: Dict[str, Tuple[int, int]],
    max_bp: int = DEFAULT_MAX_BLOCK_BP,
) -> List[Block]:
    """Split any block longer than `max_bp` into consecutive sub-blocks cut on
    that gene's own codon boundaries.

    Measured on the 2026-11-15 pilot (17 rounds, 3 genomes, evo2_7b): blocks
    of <=600bp came back with a MEDIAN OF 0 premature stop codons, blocks
    above it with a median of 10. And the failing blocks were not merely
    degraded -- their premature-stop rate was 3.8-4.5 per 100 codons against
    a chance rate of 4.7, with two blocks ABOVE chance (6.9, 11.3). That is
    the signature of frame slippage, not of a model that cannot code: Evo 2
    generates coding-like sequence but in a frame offset from the one
    BlockScorer assumes, because a single indel early in a long block
    misreads every codon after it. Shorter blocks bound how far that can
    propagate before the next real-coordinate re-anchor.

    Only single-gene blocks are split. Genes in one block can sit in
    different frames (make_blocks groups overlapping genes), so no single cut
    point is codon-aligned for all of them; a multi-gene block is returned
    whole rather than cut in a frame that is wrong for one of its members.
    16 of the pilot's 17 blocks held exactly one gene, so this covers
    essentially all of the real cases.

    Sub-blocks before the one containing a gene's true end carry that gene in
    `genes` but do NOT carry its terminal-stop expectation -- BlockScorer
    checks only for premature stops when `gene_end` lies past `block.end`.
    """
    out: List[Block] = []
    for b in blocks:
        if len(b) <= max_bp or len(b.genes) != 1:
            out.append(b)
            continue
        gs, _ge = gene_coords[b.genes[0]]
        base = max(b.start, gs)
        codon_start = base + ((3 - (base - gs) % 3) % 3)
        step = (max_bp // 3) * 3  # keep every cut on a codon boundary of this gene
        cuts = [b.start]
        pos = codon_start
        while pos + step < b.end:
            pos += step
            if pos > cuts[-1]:
                cuts.append(pos)
        cuts.append(b.end)
        for i in range(len(cuts) - 1):
            out.append(Block(b.genes, cuts[i], cuts[i + 1]))
    return out


def translate_genes(genome: str, gene_coords: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
    """Translate a target genome's own named genes in their own frame, from
    its own real coordinates. Target-agnostic -- works for phiX174 or any
    other annotated genome given its own gene_coords dict."""
    out = {}
    for name, (s, e) in gene_coords.items():
        n = (e - s) - ((e - s) % 3)
        out[name] = translate(genome[s : s + n])
    return out


def _load_default_phix174_genome() -> str:
    path = Path(__file__).parent / "phix174.fasta"
    if not path.exists():
        path = Path(__file__).parent / "dependencies" / "phix174.fasta"
    _, genome = next(read_fasta(path))
    return genome


# --------------------------------------------------------------------------
# Reference-free orthology clustering
# --------------------------------------------------------------------------

def _aa_kmer_vec(seq: str, k: int) -> np.ndarray:
    idx = kmer_indices(encode(seq), k)
    v = np.bincount(idx, minlength=_N_AA**k).astype(np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _accession(header: str) -> str:
    h = header.split("|")[-1] if "|" in header else header
    return h.split("_prot_")[0].split("_cds_")[0]


_GENE_RE = re.compile(r"\[gene=([^\]]*)\]")
_PRODUCT_RE = re.compile(r"\[protein=([^\]]*)\]")


@dataclass(frozen=True)
class CorpusRecord:
    accession: str
    gene: Optional[str]
    product: Optional[str]  # lowercased
    seq: str


def _iter_fasta_with_annotation(path: str | Path):
    """Like aa_kmer_table.read_fasta, but also keeps the rest of the NCBI
    header line (read_fasta keeps only the first whitespace-delimited
    token, which drops the [gene=...]/[protein=...] qualifiers this module
    needs). Yields (short_id, header_rest, aa_sequence)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    name, header_rest, parts = None, "", []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, header_rest, "".join(parts).upper()
                tokens = line[1:].strip().split(None, 1)
                name = tokens[0]
                header_rest = tokens[1] if len(tokens) > 1 else ""
                parts = []
            else:
                parts.append(line.strip())
    if name is not None:
        yield name, header_rest, "".join(parts).upper()


def _load_reference_corpus(
    genomes: Sequence[str | Path], exclude_accessions: Iterable[str] = ()
) -> List[CorpusRecord]:
    """CorpusRecords from one or more fasta_cds_aa files, dropping any
    record whose accession is in `exclude_accessions`."""
    excluded = set(exclude_accessions)
    corpus: List[CorpusRecord] = []
    for p in genomes:
        for short_id, header_rest, seq in _iter_fasta_with_annotation(p):
            acc = _accession(short_id)
            if acc in excluded:
                continue
            seq = seq.rstrip("*")
            if not seq:
                continue
            gene_m = _GENE_RE.search(header_rest)
            prod_m = _PRODUCT_RE.search(header_rest)
            corpus.append(CorpusRecord(
                accession=acc,
                gene=gene_m.group(1).strip() if gene_m else None,
                product=prod_m.group(1).strip().lower() if prod_m else None,
                seq=seq,
            ))
    return corpus


# Curated annotation map (dependencies/ortholog_annotation_map.json): the
# product-text-phrase -> bucket vocabulary and the gene-symbol -> bucket
# map are BOTH stated data, not inferred in code -- every phrase in it was
# hand-checked against this corpus to confirm it never matches two
# different loci within the same genome (see that file's
# "ambiguous_phrases_excluded" for ones that failed this check and were
# dropped, e.g. "minor capsid protein", reused for 2 different genes within
# NC_002643.1/NC_001741.1/NC_040350.1 alike). This is supervised assignment
# against a reviewed vocabulary, not de novo clustering -- the vocabulary
# itself is the per-family deliverable (a different target family, e.g.
# mitochondria's 13 standard-named genes, needs its own file).
_ANNOTATION_MAP_PATH = Path(__file__).parent / "dependencies" / "ortholog_annotation_map.json"


def _load_annotation_map(
    path: Path = _ANNOTATION_MAP_PATH,
) -> Tuple[Tuple[Tuple[str, str], ...], Dict[str, str], set]:
    """Returns (product_phrases, gene_symbols, reviewed_no_bucket_symbols).
    `reviewed_no_bucket_symbols` is distinct from "just not in gene_symbols"
    -- it marks a gene symbol someone has actually looked at and found no
    safe cross-genome bucket for (documented reason in the JSON file), vs a
    symbol nobody has reviewed yet. `cluster_orthologs` reports the latter
    separately so an unreviewed/unexpected symbol (e.g. a new family's own
    naming convention) is visible rather than silently falling to BLAST."""
    import json

    with open(path) as fh:
        data = json.load(fh)
    product_phrases = tuple((p, b) for p, b in data["product_phrases"])
    gene_symbols = dict(data["gene_symbols"])
    reviewed_no_bucket = set(data.get("gene_symbols_reviewed_no_bucket", {}))
    return product_phrases, gene_symbols, reviewed_no_bucket


def _product_bucket(product: Optional[str], product_phrases: Sequence[Tuple[str, str]]) -> Optional[str]:
    if not product:
        return None
    text = product.replace("putative ", "").replace("probable ", "")
    for phrase, bucket in product_phrases:
        if phrase in text:
            return bucket
    return None


def _is_nested_internal_start(rec: CorpusRecord) -> bool:
    """True for internal in-frame-start products nested entirely inside
    another gene's own ORF -- phiX174's A* is the canonical example: an
    internal start within gene A, so a truncated copy of A's own
    C-terminal region, not an independent ortholog. Left in the corpus it
    would double-count that region in A's k-mer table. Detected off the
    gene/product text (a gene symbol ending "*", or product text containing
    a bare "<letter>*" token such as "protein A*") rather than hard-coded to
    phiX174 specifically, so the same rule applies to any other nested
    product named this way. Excluded upstream of clustering entirely (not
    merged into its parent) -- see dependencies/ortholog_annotation_map.json,
    "nested_product_note"."""
    if rec.gene and rec.gene.endswith("*"):
        return True
    if rec.product and re.search(r"(?<!\w)[a-z]\*(?!\w)", rec.product):
        return True
    return False


def _exclude_nested_products(corpus: Sequence[CorpusRecord]) -> Tuple[List[CorpusRecord], List[CorpusRecord]]:
    """Drops nested/internal-start records (see `_is_nested_internal_start`)
    before they ever reach clustering or the pooled background-composition
    stats -- called explicitly from `build_tables`, not hidden inside
    `cluster_orthologs`, so both the cluster tables AND the unigram null
    model stay free of double-counted sequence. Returns (kept, excluded)."""
    kept, excluded = [], []
    for rec in corpus:
        (excluded if _is_nested_internal_start(rec) else kept).append(rec)
    return kept, excluded


@dataclass(frozen=True)
class _AnnotationResult:
    key: Optional[str]  # final tier-a/b cluster key, or None (needs BLAST)
    tier: Optional[str]  # "curated" (product text) | "symbol" (gene symbol) | None
    conflict: Optional[Tuple[str, str]]  # (symbol_bucket, product_bucket) when they disagree


def _annotation_key(
    rec: CorpusRecord, gene_symbols: Dict[str, str], product_phrases: Sequence[Tuple[str, str]]
) -> _AnnotationResult:
    """Tier a (curated product-text bucket) vs tier b (curated gene-symbol
    bucket) -- see module docstring. Precedence when a record's OWN product
    text and its gene symbol's curated bucket disagree: the gene-symbol map
    wins (it's corpus-wide curated and audited; one record's free-text
    /protein field is more likely to be the one-off annotation error) --
    every disagreement is still reported via `conflict` for `cluster_orthologs`
    to log, not silently resolved."""
    symbol_bucket = gene_symbols.get(rec.gene) if rec.gene else None
    product_bucket = _product_bucket(rec.product, product_phrases)

    if symbol_bucket and product_bucket and symbol_bucket != product_bucket:
        return _AnnotationResult(f"PRODUCT:{symbol_bucket}", "symbol", (symbol_bucket, product_bucket))
    if symbol_bucket:
        return _AnnotationResult(f"PRODUCT:{symbol_bucket}", "symbol", None)
    if product_bucket:
        return _AnnotationResult(f"PRODUCT:{product_bucket}", "curated", None)
    if rec.gene:
        return _AnnotationResult(f"SYMBOL:{rec.gene}", "symbol", None)
    return _AnnotationResult(None, None, None)


# --------------------------------------------------------------------------
# BLAST reciprocal-best-hit -- real blastp, replacing the earlier 3-mer
# cosine-similarity proxy. Used (a) to place records with no usable
# annotation and (b) as the cohesion check against annotation.
# --------------------------------------------------------------------------

def _blastp_all_vs_all(corpus: Sequence[CorpusRecord], evalue: float) -> Dict[Tuple[int, int], float]:
    """Real blastp, all sequences in `corpus` against themselves, via the
    local NCBI BLAST+ binaries in dependencies/tools/ (see
    _find_blast_bin_dir). Returns {(query_idx, hit_idx): bitscore} keeping
    only the best (highest-bitscore) row per ordered pair, self-hits
    dropped, below-`evalue` hits dropped by blastp itself via -evalue."""
    bin_dir = _find_blast_bin_dir()
    with tempfile.TemporaryDirectory(prefix="block_pipeline_blast_") as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "corpus.fasta"
        with open(fasta, "w") as fh:
            for i, rec in enumerate(corpus):
                fh.write(f">{i}\n{rec.seq}\n")
        db = tmp_path / "db"
        subprocess.run(
            [str(bin_dir / "makeblastdb.exe"), "-in", str(fasta), "-dbtype", "prot", "-out", str(db)],
            check=True, capture_output=True,
        )
        out = subprocess.run(
            [str(bin_dir / "blastp.exe"), "-query", str(fasta), "-db", str(db),
             "-evalue", str(evalue), "-outfmt", "6 qseqid sseqid bitscore"],
            check=True, capture_output=True, text=True,
        )
    best: Dict[Tuple[int, int], float] = {}
    for line in out.stdout.splitlines():
        qid, sid, bitscore = line.split("\t")
        i, j, score = int(qid), int(sid), float(bitscore)
        if i == j:
            continue
        key = (i, j)
        if key not in best or score > best[key]:
            best[key] = score
    return best


def _rbh_components(n: int, genome_of: Sequence[str], hits: Dict[Tuple[int, int], float]) -> List[int]:
    """Reciprocal-best-hit connected components: for each sequence, its best
    hit in every OTHER genome (by bitscore), kept as an edge only if
    reciprocal. Same per-genome-best + union-find shape as the original
    cosine version, generalized over any {(i, j): score} similarity map
    (higher is better). Returns a component id per index 0..n-1 (every index
    gets one, including those with no edges at all -- a singleton
    component)."""
    by_genome: Dict[str, List[int]] = {}
    for i, acc in enumerate(genome_of):
        by_genome.setdefault(acc, []).append(i)

    edges: List[Tuple[int, int]] = []
    for i in range(n):
        gi = genome_of[i]
        best_in_genome: Dict[str, int] = {}
        for (a, b), score in hits.items():
            if a != i:
                continue
            gb = genome_of[b]
            if gb == gi:
                continue
            if gb not in best_in_genome or score > hits.get((i, best_in_genome[gb]), -np.inf):
                best_in_genome[gb] = b
        for gj, j in best_in_genome.items():
            scores_back = {cand: hits.get((j, cand), -np.inf) for cand in by_genome[gi]}
            best_for_j = max(scores_back, key=scores_back.get)
            # Guard: if j has NO hit at all back into gi, every candidate
            # scores -inf and max() just returns the first one by iteration
            # order -- without this check, that can spuriously equal i and
            # fabricate a "reciprocal" edge where no real hit exists.
            if best_for_j == i and scores_back[best_for_j] > -np.inf:
                edges.append((i, j))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return [find(i) for i in range(n)]


def _cohesion_diagnostic(
    ann_key: Sequence[Optional[str]], blast_component: Sequence[int]
) -> Dict[str, Tuple[int, int]]:
    """Per annotation cluster, (members whose blast_component equals that
    cluster's most common blast_component, total members) -- "does real
    sequence similarity (BLAST, run over the WHOLE corpus, not just the
    leftovers, specifically so this check has something to compare against)
    actually support treating these as one group". Low cohesion on an
    otherwise-sensible cluster is worth a human look (see module
    docstring's NC_007856/NC_012868 "A" vs "A*" example), not something this
    pipeline tries to auto-correct."""
    by_key: Dict[str, List[int]] = {}
    for i, key in enumerate(ann_key):
        if key is not None:
            by_key.setdefault(key, []).append(i)
    report: Dict[str, Tuple[int, int]] = {}
    for key, idxs in by_key.items():
        comps = [blast_component[i] for i in idxs]
        mode_count = max((comps.count(c) for c in set(comps)), default=0)
        report[key] = (mode_count, len(idxs))
    return report


@dataclass(frozen=True)
class OrthologyDiagnostics:
    """Everything the cluster_orthologs call learned along the way, beyond
    the clusters dict itself -- printed by `_mock_test`, not consumed by
    build_tables (which only needs the clusters dict; see module
    docstring)."""
    provenance: Tuple[str, ...]  # per corpus index: "curated" | "symbol" | "blast" | "orphan"
    cohesion: Dict[str, Tuple[int, int]]  # ann cluster key -> (blast-agreeing members, total)
    tier_conflicts: Tuple[Tuple[int, str, str], ...]  # (corpus idx, symbol_bucket, product_bucket)
    unmapped_symbols: Tuple[str, ...]  # /gene values seen with no curated bucket AND not in
                                        # gene_symbols_reviewed_no_bucket -- i.e. nobody has looked yet
    n_orphans: int  # leftovers that found no annotated BLAST-component bridge
    undersized_clusters: Dict[str, int]  # cluster_key -> size, for clusters below UNDERSIZED_THRESHOLD


UNDERSIZED_THRESHOLD = 3  # a cluster this small backs off almost entirely to the pooled table anyway


def cluster_orthologs(
    corpus: Sequence[CorpusRecord], evalue: float = DEFAULT_EVALUE
) -> Tuple[Dict[str, List[int]], OrthologyDiagnostics]:
    """Ortholog clusters over `corpus`, annotation-first with BLAST for the
    leftovers -- see module docstring for the full algorithm and rationale.
    Returns ({cluster_key: [index into corpus, ...]}, OrthologyDiagnostics).
    No target genome involved; this groups the reference set among itself
    only."""
    n = len(corpus)
    product_phrases, gene_symbols, reviewed_no_bucket = _load_annotation_map()
    ann_results = [_annotation_key(rec, gene_symbols, product_phrases) for rec in corpus]
    ann_key = [r.key for r in ann_results]
    provenance: List[str] = [r.tier or "" for r in ann_results]  # "" placeholder, filled in below
    tier_conflicts = tuple((i, r.conflict[0], r.conflict[1]) for i, r in enumerate(ann_results) if r.conflict)
    unmapped_symbols = tuple(sorted({
        rec.gene for rec in corpus
        if rec.gene and rec.gene not in gene_symbols and rec.gene not in reviewed_no_bucket
    }))

    # BLAST runs over the WHOLE corpus (not pre-filtered to leftovers): it
    # both bridges unannotated records into an existing annotation cluster
    # AND, on the already-annotated records, gives the cohesion diagnostic
    # above something real to compare against.
    genome_of = [rec.accession for rec in corpus]
    hits = _blastp_all_vs_all(corpus, evalue)
    blast_component = _rbh_components(n, genome_of, hits)

    # Bridge leftovers (ann_key is None) via shared BLAST component with an
    # annotated record; conflicting annotated members in the same component
    # are left alone (annotation wins, never auto-merged) and the leftover
    # stays its own BLAST-only cluster ("orphan") rather than guessing which
    # side it's on.
    component_ann_keys: Dict[int, set] = {}
    for i, comp in enumerate(blast_component):
        if ann_key[i] is not None:
            component_ann_keys.setdefault(comp, set()).add(ann_key[i])

    final_key: List[str] = []
    n_orphans = 0
    for i in range(n):
        if ann_key[i] is not None:
            final_key.append(ann_key[i])
            continue
        candidates = component_ann_keys.get(blast_component[i], set())
        if len(candidates) == 1:
            final_key.append(next(iter(candidates)))
            provenance[i] = "blast"
        else:
            final_key.append(f"BLAST:{blast_component[i]}")
            provenance[i] = "orphan"
            n_orphans += 1

    clusters: Dict[str, List[int]] = {}
    for i, key in enumerate(final_key):
        clusters.setdefault(key, []).append(i)

    undersized = {key: len(idxs) for key, idxs in clusters.items() if len(idxs) < UNDERSIZED_THRESHOLD}
    cohesion = _cohesion_diagnostic(ann_key, blast_component)
    diagnostics = OrthologyDiagnostics(
        provenance=tuple(provenance),
        cohesion=cohesion,
        tier_conflicts=tier_conflicts,
        unmapped_symbols=unmapped_symbols,
        n_orphans=n_orphans,
        undersized_clusters=undersized,
    )
    return clusters, diagnostics


# --------------------------------------------------------------------------
# build_tables + per-target ortholog assignment
# --------------------------------------------------------------------------

def build_tables(
    genomes: Sequence[str | Path],
    target_excluded: bool = True,
    target_accession: str = PHIX174_ACCESSION,
    ks: Sequence[int] = DEFAULT_KS,
    mu: float = DEFAULT_MU,
    evalue: float = DEFAULT_EVALUE,
) -> Dict[str, AAKmerTable]:
    """Reference-free per-cluster amino-acid k-mer tables.

    genomes: CDS-translation FASTA path(s) (fasta_cds_aa format -- the same
        files build_aa_kmer_table.py already consumes). Target-agnostic:
        for Target A this is the 20-genome Microviridae reference set; for
        Target B it is the family minus whichever genome is held out this
        round. Same function, different list.
    target_excluded: if True (default; diagnostic 4's "held out" case),
        drop any record whose accession matches `target_accession`. Set
        False only to run diagnostic 4's with-vs-without leak-size
        comparison (requires the target's own CDS fasta to be present in
        `genomes` in the first place).
    target_accession: which accession counts as "the target" for the
        exclusion above. Defaults to phiX174's for Target A; Target B
        passes whichever genome is held out that round.

    Returns {cluster_id: AAKmerTable}. Each table carries `.null_freq`
    (unigram amino-acid composition, shared across every cluster) and
    `.centroid_vec` (mean normalized amino-acid 3-mer profile of the
    cluster's members -- the only thing assign_target_genes() needs to map
    a target's own genes onto these clusters) and `.n_group_sequences`
    (diagnostics only).
    """
    excluded = {target_accession} if target_excluded else set()
    corpus = _load_reference_corpus(genomes, excluded)
    corpus, nested_excluded = _exclude_nested_products(corpus)
    if nested_excluded:
        print(f"build_tables: excluded {len(nested_excluded)} nested/internal-start product(s) "
              f"before clustering (not merged into their parent gene): "
              + ", ".join(f"{r.accession}:{r.gene or r.product}" for r in nested_excluded))
    clusters, diagnostics = cluster_orthologs(corpus, evalue=evalue)
    if diagnostics.tier_conflicts:
        print(f"build_tables: {len(diagnostics.tier_conflicts)} annotation tier conflict(s) "
              f"(gene-symbol bucket vs this record's own /protein text disagreed; symbol wins) -- "
              + "; ".join(f"corpus[{i}] symbol={sb!r} product={pb!r}" for i, sb, pb in diagnostics.tier_conflicts))
    if diagnostics.unmapped_symbols:
        print(f"build_tables: unreviewed /gene symbols with no curated bucket (falling through to "
              f"BLAST): {diagnostics.unmapped_symbols}")
    if diagnostics.undersized_clusters:
        print(f"build_tables: {len(diagnostics.undersized_clusters)} cluster(s) below "
              f"UNDERSIZED_THRESHOLD={UNDERSIZED_THRESHOLD} members (smoothing dominates these): "
              + ", ".join(f"{k}={n}" for k, n in sorted(diagnostics.undersized_clusters.items())))
    print(f"build_tables: {diagnostics.n_orphans} orphan record(s) (no annotation, no BLAST bridge "
          f"to an annotated cluster)")
    low_cohesion = {k: v for k, v in diagnostics.cohesion.items() if v[0] < v[1]}
    print(f"build_tables: BLAST cohesion check (does real sequence similarity, over the whole "
          f"corpus, agree with annotation?) on {len(diagnostics.cohesion)} annotated clusters, "
          f"{len(low_cohesion)} with a member outside the cluster's dominant BLAST component:")
    for key, (agree, total) in sorted(low_cohesion.items()):
        print(f"    {key}: {agree}/{total} agree")

    pooled = AAKmerTable(ks)
    unigram_counts = np.zeros(_N_AA, dtype=np.float64)
    for rec in corpus:
        pooled.add_sequence(rec.seq)
        codes = encode(rec.seq)
        unigram_counts += np.bincount(codes[codes >= 0], minlength=_N_AA)
    pooled.finalize()
    total = unigram_counts.sum()
    null_freq = (unigram_counts / total) if total > 0 else np.full(_N_AA, 1.0 / _N_AA)

    tables: Dict[str, AAKmerTable] = {}
    for cid, indices in clusters.items():
        member_seqs = [corpus[i].seq for i in indices]
        member_accessions = [corpus[i].accession for i in indices]
        t = AAKmerTable(ks)
        for s in member_seqs:
            t.add_sequence(s)
        for k in ks:
            backed = t.counts[k] + mu * pooled.probs[k]
            denom = backed.sum()
            t.probs[k] = backed / denom if denom > 0 else backed
        centroid = sum(_aa_kmer_vec(s, k=3) for s in member_seqs)
        cnorm = np.linalg.norm(centroid)
        t.centroid_vec = centroid / cnorm if cnorm > 0 else centroid
        t.null_freq = null_freq
        t.n_group_sequences = len(member_seqs)
        t.member_seqs = member_seqs  # for assign_target_genes' BLAST fallback
        t.member_accessions = member_accessions
        tables[cid] = t

    # Full size distribution + implied backoff weight (n_letters/(n_letters+mu),
    # how much of each cluster's probability mass is its own real counts vs
    # the pooled backoff) -- the decision table's real rule is on the MEDIAN
    # cluster size (>=15 -> per-gene-with-backoff viable; many below 8 ->
    # pooled), not the coarse UNDERSIZED_THRESHOLD cutoff above, which only
    # flags the extreme tail.
    sizes = sorted(t.n_group_sequences for t in tables.values())
    median_size = float(np.median(sizes)) if sizes else 0.0
    print(f"build_tables: cluster size distribution (n={len(sizes)} clusters): {sizes}")
    print(f"build_tables: median cluster size={median_size:.1f} -- decision-gate rule: "
          f">=15 means per-gene-with-backoff is viable; many clusters <8 means fall back to pooled "
          f"(currently {'PER-GENE VIABLE' if median_size >= 15 else 'BELOW the per-gene threshold'})")
    for cid in sorted(tables, key=lambda c: tables[c].n_group_sequences):
        t = tables[cid]
        weight = t.n_letters / (t.n_letters + mu) if (t.n_letters + mu) > 0 else 0.0
        print(f"    {cid}: n_sequences={t.n_group_sequences} n_letters={t.n_letters} "
              f"own_weight~{weight:.3f} (vs pooled backoff {1 - weight:.3f})")
    return tables


DEFAULT_MU_SWEEP: Tuple[float, ...] = (250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0,
                                        2000.0, 2500.0, 3000.0, 4000.0, 6000.0, 8000.0)


def _log_likelihood_with_backoff(
    seq: str, counts: Dict[int, np.ndarray], pooled_probs: Dict[int, np.ndarray],
    mu: float, ks: Sequence[int]
) -> Tuple[float, int]:
    """Real per-k-mer-window log-likelihood under a backed-off model for
    one specific mu, built from a counts dict rather than a fixed table --
    lets `sweep_mu` try many mu values against the SAME cluster counts
    without rebuilding anything. Deliberately NOT AAKmerTable.score_aa's
    "relative" mode (P_k * 20**k, equal weight per k) -- that's a ratio
    built for ranking candidates during generation, not a proper scoring
    rule for model selection, and sweeping mu against it came back
    monotonically favoring less smoothing all the way down to the sweep's
    floor with no interior maximum -- the opposite of what a real
    log-likelihood should do here (too little smoothing should get
    punished by the held-out gene's rarer k-mers landing on near-zero
    backed probability). Returns (summed log-likelihood, number of k-mer
    windows) so the caller can average correctly across many genes/k's
    rather than averaging an average."""
    seq = seq.rstrip("*")
    total_ll = 0.0
    total_n = 0
    for k in ks:
        backed = counts[k] + mu * pooled_probs[k]
        denom = backed.sum()
        probs = backed / denom if denom > 0 else backed
        idx = kmer_indices(encode(seq), k)
        if len(idx) == 0:
            continue
        total_ll += float(np.log(probs[idx] + 1e-12).sum())
        total_n += len(idx)
    return total_ll, total_n


def sweep_mu(
    genomes: Sequence[str | Path],
    mu_values: Sequence[float] = DEFAULT_MU_SWEEP,
    ks: Sequence[int] = DEFAULT_KS,
    evalue: float = DEFAULT_EVALUE,
) -> Dict[float, float]:
    """Leave-one-genome-out likelihood sweep for the backoff strength mu --
    decision table, "Smoothing strength (mu)" row: "Sweep 250 to 8,000,
    take the maximum; expect 1000-2500". For each of the 20 reference
    genomes in turn: hold it out completely, run the real annotation+BLAST
    `cluster_orthologs` on the remaining 19 (mu-independent -- run ONCE per
    held-out genome, not once per (genome, mu) pair, since only the
    backoff step below depends on mu), then score the held-out genome's
    OWN real CDS translations against each cluster's counts backed off at
    every candidate mu. A held-out gene only contributes if ITS OWN
    annotation (same curated map `cluster_orthologs` uses) names a cluster
    that exists in the other 19 genomes -- genes with no usable annotation
    are skipped here (this sweep is about tuning the backoff for the
    clusters the method actually trusts, not re-deriving BLAST-bridging
    for every held-out gene 20x per mu).

    Returns {mu: mean per-k-mer-window log-likelihood, averaged over every
    k-mer window of every scored held-out gene across all 20 genomes and
    both k=2,3}. Pick the mu with the highest (least negative) value."""
    full_corpus = _load_reference_corpus(genomes)
    full_corpus, _ = _exclude_nested_products(full_corpus)
    accessions = sorted({rec.accession for rec in full_corpus})
    product_phrases, gene_symbols, _reviewed = _load_annotation_map()

    sums = {mu: 0.0 for mu in mu_values}
    weights = {mu: 0 for mu in mu_values}

    for held_out in accessions:
        corpus = _load_reference_corpus(genomes, exclude_accessions={held_out})
        corpus, _ = _exclude_nested_products(corpus)
        clusters, _diag = cluster_orthologs(corpus, evalue=evalue)

        pooled = AAKmerTable(ks)
        for rec in corpus:
            pooled.add_sequence(rec.seq)
        pooled.finalize()

        cluster_counts: Dict[str, Dict[int, np.ndarray]] = {}
        for cid, indices in clusters.items():
            t = AAKmerTable(ks)
            for i in indices:
                t.add_sequence(corpus[i].seq)
            cluster_counts[cid] = t.counts

        for rec in (r for r in full_corpus if r.accession == held_out):
            ann = _annotation_key(rec, gene_symbols, product_phrases)
            cid = ann.key
            if cid is None or cid not in cluster_counts:
                continue
            for mu in mu_values:
                ll, n = _log_likelihood_with_backoff(rec.seq, cluster_counts[cid], pooled.probs, mu, ks)
                sums[mu] += ll
                weights[mu] += n

    return {mu: (sums[mu] / weights[mu] if weights[mu] else float("nan")) for mu in mu_values}


def _blast_bridge_target_gene(
    seq: str, tables: Dict[str, AAKmerTable], evalue: float
) -> Optional[str]:
    """Fallback for a target gene whose symbol has no curated bucket: blast
    it against the WHOLE reference corpus (reconstructed from each table's
    retained `member_seqs`/`member_accessions`) and bridge into whichever
    existing cluster its reciprocal-best-hit component contains -- the same
    bridging rule `cluster_orthologs` itself uses for unannotated
    leftovers, just run once more with this one target gene added. Returns
    None (caller falls back to cosine-nearest-centroid) if no single
    cluster is reciprocally confirmed."""
    flat_accessions: List[str] = []
    flat_seqs: List[str] = []
    flat_cluster_key: List[str] = []
    for key, table in tables.items():
        for acc, s in zip(table.member_accessions, table.member_seqs):
            flat_accessions.append(acc)
            flat_seqs.append(s)
            flat_cluster_key.append(key)
    flat_accessions.append("__TARGET__")
    flat_seqs.append(seq)

    records = [CorpusRecord(accession=a, gene=None, product=None, seq=s)
               for a, s in zip(flat_accessions, flat_seqs)]
    hits = _blastp_all_vs_all(records, evalue)
    components = _rbh_components(len(records), flat_accessions, hits)
    target_component = components[-1]
    candidates = {flat_cluster_key[i] for i in range(len(records) - 1) if components[i] == target_component}
    return next(iter(candidates)) if len(candidates) == 1 else None


def assign_target_genes(
    gene_seqs: Dict[str, str], tables: Dict[str, AAKmerTable], evalue: float = DEFAULT_EVALUE
) -> Dict[str, str]:
    """Map a specific target genome's own named genes onto the existing
    clusters. Annotation first, same principle as cluster_orthologs: if the
    target's own gene name is a symbol the curated map
    (dependencies/ortholog_annotation_map.json) has a bucket for, use that
    cluster directly -- phiX174's own gene letters (C, D, E, F, G, H, J) are
    already keys in that map, so this is a real lookup, not a guess.

    For a gene with no curated symbol (phiX174's K, here), fall back to a
    real BLAST reciprocal-best-hit bridge (`_blast_bridge_target_gene`),
    same mechanism cluster_orthologs itself uses for unannotated leftovers.
    Only if THAT also finds no single reciprocally-confirmed cluster does
    this drop to nearest-centroid cosine similarity (3-mer amino-acid
    composition) as the last resort.

    Composition-cosine alone is NOT reliable here: it was internally
    consistent when clusters were themselves formed by that same
    composition metric (the old cluster_orthologs), but clusters are now
    formed by annotation + BLAST, a different signal composition-cosine has
    no reason to agree with -- discovered when it silently assigned two
    different phiX174 genes to the same cluster before this fallback chain
    existed. The target genome still plays no part in DEFINING the clusters
    either way -- it is only ever looked up against them here, which is
    what keeps this step identical for Target A (phiX174) and Target B (any
    held-out family member): same function, just called with that target's
    own translated genes and gene symbols."""
    _phrases, gene_symbols, _reviewed = _load_annotation_map()
    assignment: Dict[str, str] = {}
    for name, seq in gene_seqs.items():
        curated_key = f"PRODUCT:{gene_symbols[name]}" if name in gene_symbols else None
        if curated_key is not None and curated_key in tables:
            assignment[name] = curated_key
            continue

        bridged = _blast_bridge_target_gene(seq, tables, evalue)
        if bridged is not None:
            assignment[name] = bridged
            continue

        v = _aa_kmer_vec(seq, k=3)
        best_cid, best_sim = None, -1.0
        for cid, table in tables.items():
            s = float(v @ table.centroid_vec)
            if s > best_sim:
                best_cid, best_sim = cid, s
        assignment[name] = best_cid
    return assignment


@dataclass(frozen=True)
class CoverageReport:
    """Per-genome and overall breakdown of how a new candidate target
    genome's own genes would resolve against an EXISTING corpus's tables,
    without building anything new -- answers "is there even a table to
    score this genome against" before spending any time expanding the
    corpus. curated: the gene's OWN /gene or /protein annotation matches
    the curated map directly (same tier-a/b lookup cluster_orthologs uses,
    not assign_target_genes' name-only lookup -- a divergent target genome
    won't usually share phiX174's literal gene letters, so this checks its
    OWN product text too). blast: no curated match, but a real BLAST RBH
    bridges it into an existing cluster. fallthrough: neither -- the only
    thing left is cosine-nearest-centroid, the one step already shown to
    be unreliable (collided two different phiX174 genes once)."""
    per_accession: Dict[str, Dict[str, int]]  # accession -> {curated, blast, fallthrough, total}
    totals: Dict[str, int]


def _resolve_target_gene(
    rec: CorpusRecord, tables: Dict[str, AAKmerTable],
    gene_symbols: Dict[str, str], product_phrases: Sequence[Tuple[str, str]], evalue: float,
) -> Tuple[Optional[str], str]:
    """Shared resolution path for `target_coverage` and `prepare_target_b`:
    does this target gene's OWN annotation (product text or gene symbol --
    the same tier-a/b check `cluster_orthologs` uses, not a name-only
    lookup) match an existing cluster; if not, does a real BLAST RBH bridge
    it into one. Returns (cluster_key_or_None, provenance) where provenance
    is "curated" | "blast" | "fallthrough". A None key means the caller
    should fall back to cosine-nearest-centroid as the last resort (see
    assign_target_genes) -- this function never does that itself, so
    target_coverage can count a true fallthrough instead of silently
    picking something via the unreliable path."""
    ann = _annotation_key(rec, gene_symbols, product_phrases)
    if ann.key is not None and ann.key in tables:
        return ann.key, "curated"
    bridged = _blast_bridge_target_gene(rec.seq, tables, evalue)
    if bridged is not None:
        return bridged, "blast"
    return None, "fallthrough"


def target_coverage(
    target_corpus: Sequence[CorpusRecord], tables: Dict[str, AAKmerTable], evalue: float = DEFAULT_EVALUE
) -> CoverageReport:
    """Gene-content coverage check for a candidate new target set against
    an EXISTING corpus's tables -- see CoverageReport. Does not build or
    change anything; purely diagnostic, meant to run BEFORE deciding
    whether a target set needs its own corpus expansion (decision table's
    spirit, applied one level up: "does the data support the design" here
    means "do these targets' genes even have a cluster to be scored
    against", not just "is the per-gene-vs-pooled split viable")."""
    product_phrases, gene_symbols, _reviewed = _load_annotation_map()
    per_accession: Dict[str, Dict[str, int]] = {}
    for rec in target_corpus:
        bucket = per_accession.setdefault(rec.accession, {"curated": 0, "blast": 0, "fallthrough": 0, "total": 0})
        bucket["total"] += 1
        _key, provenance = _resolve_target_gene(rec, tables, gene_symbols, product_phrases, evalue)
        bucket[provenance] += 1

    totals = {"curated": 0, "blast": 0, "fallthrough": 0, "total": 0}
    for bucket in per_accession.values():
        for key in totals:
            totals[key] += bucket[key]
    return CoverageReport(per_accession=per_accession, totals=totals)


# --------------------------------------------------------------------------
# score_candidate: translate, gate, PMI, mean
# --------------------------------------------------------------------------

def _gene_frame_aa(dna_block: str, block_start: int, gene_start: int, gene_end: int) -> Tuple[str, int]:
    """Translate `dna_block` (covering [block_start, block_start+len)) in
    one gene's own frame. Returns (aa_seq, stop_idx): stop_idx is the
    0-based amino-acid index that should hold that gene's real terminal
    stop codon. By construction (make_blocks always extends a block to the
    end of its longest gene) gene_end always falls inside dna_block, so
    stop_idx is always well-defined here -- a ValueError means a Block was
    built from a gene_coords dict make_blocks didn't actually produce."""
    base = max(block_start, gene_start)
    codon_start = base + ((3 - (base - gene_start) % 3) % 3)
    rel_start = codon_start - block_start
    if rel_start < 0 or rel_start > len(dna_block):
        raise ValueError("gene frame starts outside this block's dna -- block/gene mismatch")
    aa = translate(dna_block[rel_start:])
    stop_idx = (gene_end - codon_start) // 3 - 1
    if stop_idx < 0 or stop_idx >= len(aa):
        raise ValueError("gene's real stop codon falls outside this block's dna -- block/gene mismatch")
    return aa, stop_idx


def _null_kmer_table(null_freq: np.ndarray, k: int) -> np.ndarray:
    """Flat length-20**k array of P_null for every k-mer, aligned with
    kmer_indices' base-20 encoding (first residue most significant). The
    k-fold outer product of the unigram frequency vector with itself,
    raveled, is exactly that -- see module docstring, "PMI scoring"."""
    arr = null_freq
    for _ in range(k - 1):
        arr = np.multiply.outer(arr, null_freq)
    return arr.ravel()


class BlockScorer:
    """Binds a set of cluster tables (from build_tables) plus one target's
    own gene->cluster assignment (from assign_target_genes) and gene
    coordinates to a scorer with the test-plan's exact
    score_candidate(dna, block) -> float signature. Everything target-
    specific is confined to `gene_to_cluster` and `gene_coords`; the
    scoring math itself (translate, gate, PMI, mean) never branches on
    which target it's running for."""

    def __init__(
        self,
        tables: Dict[str, AAKmerTable],
        gene_to_cluster: Dict[str, str],
        gene_coords: Dict[str, Tuple[int, int]],
    ):
        self.tables = tables
        self.gene_to_cluster = gene_to_cluster
        self.gene_coords = gene_coords
        any_table = next(iter(tables.values()))
        self.ks = any_table.ks
        self.null_kmer_probs = {k: _null_kmer_table(any_table.null_freq, k) for k in self.ks}

    def _pmi(self, region: str, table: AAKmerTable) -> Optional[float]:
        """Mean PMI (log-ratio of observed k-mer probability over the
        amino-acid-composition null) over `region`'s k-mer windows. Split on
        "*" so a candidate that DOES carry premature stops still earns a real
        composition score from its stop-free stretches, rather than either
        scoring nothing or fabricating k-mer windows that span a stop codon.
        Returns None when no window is long enough to score."""
        window_scores: List[np.ndarray] = []
        for segment in region.split("*"):
            for k in self.ks:
                if len(segment) < k:
                    continue
                idx = kmer_indices(encode(segment), k)
                if len(idx) == 0:
                    continue
                p_obs = table.probs[k][idx]
                p_null = self.null_kmer_probs[k][idx]
                window_scores.append(np.log(p_obs + 1e-12) - np.log(p_null + 1e-12))
        if not window_scores:
            return None
        return float(np.concatenate(window_scores).mean())

    def score_candidate_detailed(self, dna: str, block: Block) -> Tuple[float, int]:
        """(score, n_violations) for one candidate over `block`.

        A violation is one premature stop codon inside a gene's coding
        region, or that gene's terminal stop codon being absent where its
        real coordinates say it belongs. Candidates with zero violations are
        scored on PMI alone and always outrank any candidate with a
        violation (see VIOLATION_PENALTY); among violators, fewer violations
        wins and PMI breaks the tie.

        Replaces a flat `return -inf` hard gate (pre-2026-11-14). That gate
        was unsatisfiable at block scale -- it demanded no stop codon across
        hundreds of residues AND one at exactly the right index, roughly
        3e-12 by chance for a 553-residue gene -- so on a real run it
        rejected 98/98 rounds across all 23 genomes. Because every candidate
        then scored an identical -inf, `max()` returned index 0 and the
        caller appended candidate 0 every single round: generation still
        produced complete, contiguous sequence, so the run LOOKED fine while
        no k-mer selection had occurred anywhere in it. Grading by violation
        count is what generate_block's own docstring already called for --
        "rank an all-broken round by fewest violations rather than discard it
        silently" -- and it keeps a fully-broken round informative instead of
        collapsing it into a silent tie."""
        frame_scores: List[float] = []
        n_premature = 0
        n_terminal = 0
        for g in block.genes:
            gs, ge = self.gene_coords[g]
            base = max(block.start, gs)
            codon_start = base + ((3 - (base - gs) % 3) % 3)
            rel_start = codon_start - block.start
            if rel_start >= len(dna):
                continue  # none of this gene falls in this sub-block (leading spacer)
            aa = translate(dna[rel_start:])
            if ge <= block.end:
                # This block covers the gene's real end, so its terminal stop
                # codon belongs inside it at a known index.
                stop_idx = (ge - codon_start) // 3 - 1
                if stop_idx < 0 or stop_idx >= len(aa):
                    raise ValueError(
                        "gene's real stop codon falls outside this block's dna -- block/gene mismatch"
                    )
                region = aa[:stop_idx]
                if aa[stop_idx] != "*":
                    n_terminal += 1
            else:
                # Sub-block from split_long_blocks: the gene runs on past this
                # block, so there is no terminal stop to expect here and the
                # whole translated stretch should be stop-free.
                region = aa
            n_premature += region.count("*")
            if not region:
                continue
            pmi = self._pmi(region, self.tables[self.gene_to_cluster[g]])
            if pmi is not None:
                frame_scores.append(pmi)
        pmi_mean = float(np.mean(frame_scores)) if frame_scores else 0.0
        violations = n_premature + n_terminal
        self.last_breakdown = (n_premature, n_terminal)
        return pmi_mean - VIOLATION_PENALTY * violations, violations

    def score_candidate_breakdown(self, dna: str, block: Block) -> Tuple[float, int, int]:
        """(score, n_premature, n_terminal). The split matters: on the
        2026-11-15 pilot, 9 of 17 rounds had ZERO premature stops and failed
        ONLY on terminal-stop placement, while 16 of 17 missed the terminal
        stop. A single `n_violations` count hid that, and the two numbers call
        for completely different fixes -- forcing the boundary stop codon
        versus bounding frame slippage with smaller blocks."""
        score, _violations = self.score_candidate_detailed(dna, block)
        n_prem, n_term = self.last_breakdown
        return score, n_prem, n_term

    def score_candidate(self, dna: str, block: Block) -> float:
        """The test plan's exact score_candidate(dna, block) -> float
        signature. See score_candidate_detailed for the scoring rule and for
        why this no longer returns -inf."""
        return self.score_candidate_detailed(dna, block)[0]


# --------------------------------------------------------------------------
# generate_block: batched, no verification
# --------------------------------------------------------------------------

def generate_block(
    prefix: str,
    block: Block,
    c: int,
    generate_fn: Callable[[str, int, int], str],
    base_seed: int = 0,
) -> List[str]:
    """c complete candidates for `block`, each generate_fn(prefix, n_tokens,
    seed) call producing len(block) new bases continuing from `prefix`.

    "Batched" per the test plan means c candidates from one logical call
    site, not c sequential decisions each conditioned on the others -- NOT
    yet a single multi-sequence forward pass: EvSpark's generate_native is
    currently single-sequence only (EvSpark/evspark/api.py, generate_native/
    native_sample_reference hard-code batch=1). This loops c seeds for now,
    same shape as kmer_fulllength_benchmark.py's existing round loop, so
    swapping in a real batched call later (once EvSpark gains it -- compute
    budget section, "batch the candidates") is a one-function change here,
    not a pipeline rewrite.

    No verification: candidates are NOT filtered for premature stops here.
    That gate lives in BlockScorer.score_candidate, which needs to see
    broken candidates too (e.g. to rank an all-broken round by fewest
    violations rather than discard it silently)."""
    n = len(block)
    seeds = [int(base_seed * 1_000_003 + block.start * 1009 + j) for j in range(c)]
    return [generate_fn(prefix, n, s) for s in seeds]


# --------------------------------------------------------------------------
# Target A convenience wrapper -- the only phiX174-specific code in this
# file, and it's pure glue: every function it calls is target-agnostic.
# Target B's equivalent wrapper would call the same four functions with a
# different genome, gene_coords dict, and held-out accession.
# --------------------------------------------------------------------------

def prepare_target_a(
    genomes: Sequence[str | Path],
    phix_genome: str | None = None,
    **build_kwargs,
) -> Tuple[Dict[str, AAKmerTable], Dict[str, str], List[Block]]:
    if phix_genome is None:
        phix_genome = _load_default_phix174_genome()
    tables = build_tables(genomes, target_excluded=True, target_accession=PHIX174_ACCESSION, **build_kwargs)
    gene_seqs = translate_genes(phix_genome, PHIX174_GENES)
    gene_to_cluster = assign_target_genes(gene_seqs, tables, evalue=build_kwargs.get("evalue", DEFAULT_EVALUE))
    blocks = make_blocks(PHIX174_GENES)
    return tables, gene_to_cluster, blocks


# --------------------------------------------------------------------------
# Target B convenience wrapper -- generic, for any target genome other than
# phiX174. Real target annotation brings two complications phiX174 never
# has (it is entirely +1 strand and its only origin-wrapping genes, A/A*/B,
# are already excluded from PHIX174_GENES by this file's own existing
# convention); handled below the same way: drop what isn't supported
# rather than guess.
# --------------------------------------------------------------------------

_LOCATION_RE = re.compile(r"^(complement\()?(\d+)\.\.(\d+)\)?$")
_LOCATION_FIELD_RE = re.compile(r"\[location=([^\]]+)\]")
_PROTEIN_ID_RE = re.compile(r"\[protein_id=([^\]]+)\]")


def _parse_location(loc: str) -> Optional[Tuple[int, int, bool]]:
    """NCBI fasta_cds_aa [location=...] -> (start0, end_halfopen,
    is_complement), 1-based inclusive converted to 0-based half-open.
    Returns None for anything this parses can't handle cleanly -- any
    join() location (an origin-wrapping gene, or in one observed real case
    -- PQ006826.1 -- a non-origin internal gap that isn't a wrap at all)."""
    if "join" in loc:
        return None
    m = _LOCATION_RE.match(loc)
    if not m:
        return None
    is_complement = m.group(1) is not None
    start_1based, end_1based = int(m.group(2)), int(m.group(3))
    return start_1based - 1, end_1based, is_complement


@dataclass(frozen=True)
class _TargetGeneSpec:
    name: str
    start: int
    end: int
    is_complement: bool
    product: Optional[str]
    aa_seq: str  # NCBI's own translation -- already correctly stranded, reused as-is for resolution


def _load_target_gene_specs(
    cds_fasta_path: str | Path, accession: str
) -> Tuple[List[_TargetGeneSpec], int]:
    """One target accession's CDS records from a fasta_cds_aa file, parsed
    for (name, coords, strand, product, translation). Returns (specs,
    n_dropped) where n_dropped counts join()-location records skipped
    (see _parse_location)."""
    specs: List[_TargetGeneSpec] = []
    n_dropped = 0
    for short_id, header_rest, seq in _iter_fasta_with_annotation(cds_fasta_path):
        if _accession(short_id) != accession:
            continue
        loc_m = _LOCATION_FIELD_RE.search(header_rest)
        if not loc_m:
            continue
        parsed = _parse_location(loc_m.group(1))
        if parsed is None:
            n_dropped += 1
            continue
        start, end, is_complement = parsed
        prod_m = _PRODUCT_RE.search(header_rest)
        pid_m = _PROTEIN_ID_RE.search(header_rest)
        name = pid_m.group(1) if pid_m else short_id
        product = prod_m.group(1).strip().lower() if prod_m else None
        specs.append(_TargetGeneSpec(
            name=name, start=start, end=end, is_complement=is_complement,
            product=product, aa_seq=seq.rstrip("*"),
        ))
    return specs, n_dropped


def prepare_target_b(
    accession: str,
    genome: str,
    cds_fasta_path: str | Path,
    tables: Dict[str, AAKmerTable],
    evalue: float = DEFAULT_EVALUE,
) -> Tuple[Dict[str, str], Dict[str, Tuple[int, int]], List[Block], str]:
    """Generic per-target prep for any target genome OTHER than phiX174 --
    what prepare_target_a is for phiX174 specifically. Uses the EXISTING
    `tables` (built once from the 20-genome reference corpus) unchanged --
    same invariant as prepare_target_a; this function only ever looks
    tables up, never rebuilds them.

    Returns (gene_to_cluster, gene_coords, blocks, genome) -- `genome` is
    returned because it may be reverse-complemented from the input (see
    below); callers MUST use the returned genome with the returned
    gene_coords, not the original, or coordinates won't line up.

    Real annotation brings two complications phiX174 never has:
      - join()-location genes are dropped individually (same policy
        PHIX174_GENES already applies to A/A*/B).
      - If EVERY gene on this accession is on the minus strand (a
        submitter's arbitrary choice of which strand to deposit, not a
        biological anomaly -- 3 of the 28 genomes checked this session
        are entirely minus-strand), the whole genome is reverse-
        complemented once and coordinates remapped, turning it into an
        ordinary forward-strand problem without touching
        translate_genes/_gene_frame_aa/BlockScorer at all. A genome with a
        genuine MIX of strands (not observed in this session's 28) has
        its minus-strand genes dropped instead -- true mixed-strand
        handling isn't built."""
    specs, n_dropped_join = _load_target_gene_specs(cds_fasta_path, accession)
    if n_dropped_join:
        print(f"prepare_target_b({accession}): dropped {n_dropped_join} join()-location gene(s)")

    all_complement = bool(specs) and all(s.is_complement for s in specs)
    if all_complement:
        L = len(genome)
        genome = revcomp(genome)
        specs = [_TargetGeneSpec(s.name, L - s.end, L - s.start, False, s.product, s.aa_seq) for s in specs]
        print(f"prepare_target_b({accession}): entire annotation was minus-strand -- "
              f"reverse-complemented the genome and remapped coordinates")
    else:
        mixed = [s for s in specs if s.is_complement]
        if mixed:
            print(f"prepare_target_b({accession}): dropped {len(mixed)} minus-strand gene(s) "
                  f"on an otherwise-forward-strand genome -- mixed-strand handling isn't built")
            specs = [s for s in specs if not s.is_complement]

    product_phrases, gene_symbols, _reviewed = _load_annotation_map()
    gene_coords: Dict[str, Tuple[int, int]] = {}
    gene_to_cluster: Dict[str, str] = {}
    for s in specs:
        gene_coords[s.name] = (s.start, s.end)
        rec = CorpusRecord(accession=accession, gene=None, product=s.product, seq=s.aa_seq)
        key, provenance = _resolve_target_gene(rec, tables, gene_symbols, product_phrases, evalue)
        if key is None:
            v = _aa_kmer_vec(s.aa_seq, k=3)
            best_cid, best_sim = None, -1.0
            for cid, table in tables.items():
                sim = float(v @ table.centroid_vec)
                if sim > best_sim:
                    best_cid, best_sim = cid, sim
            key, provenance = best_cid, "cosine-fallback"
        gene_to_cluster[s.name] = key

    blocks = make_blocks(gene_coords)
    return gene_to_cluster, gene_coords, blocks, genome


# --------------------------------------------------------------------------
# CPU-only mock self-test (no GPU, $0) -- exercises build_tables,
# assign_target_genes, BlockScorer, and generate_block end to end against
# the real reference data and real phiX174 genome, with a fake sampler
# standing in for Evo2.
# --------------------------------------------------------------------------

def _mock_test() -> None:
    from aa_kmer_table import CODON_TABLE

    default_fasta = Path(__file__).parent / "dependencies" / "microviridae_refs_cds_aa.fasta"
    genome = _load_default_phix174_genome()

    # Pins PHIX174_GENES' labelling: a span attached to the wrong gene name
    # translates to the WRONG LENGTH protein (the real spans are correct,
    # only the names were ever wrong) -- see EXPECTED_RESIDUE_LENGTHS.
    gene_seqs = translate_genes(genome, PHIX174_GENES)
    for name, aa in gene_seqs.items():
        stop_idx = aa.find("*")
        body_len = stop_idx if stop_idx != -1 else len(aa)
        expected = EXPECTED_RESIDUE_LENGTHS[name]
        assert body_len == expected, (
            f"BUG: gene {name!r} at {PHIX174_GENES[name]} translates to {body_len} residues, "
            f"expected {expected} -- PHIX174_GENES has the wrong name on this span"
        )

    tables, gene_to_cluster, blocks = prepare_target_a([default_fasta])

    print(f"reference-free clustering: {len(tables)} clusters off the corpus "
          f"(diagnostic-2-adjacent -- does the table discriminate at all, cheaply)")
    print("target gene -> cluster assignment:")
    for g in sorted(gene_to_cluster, key=lambda g: PHIX174_GENES[g][0]):
        cid = gene_to_cluster[g]
        t = tables[cid]
        print(f"  {g} -> cluster {cid}  n_sequences={t.n_group_sequences} n_letters={t.n_letters}")

    # Genes with a curated symbol in the annotation map are guaranteed
    # distinct buckets by construction of that map (each letter was hand-
    # assigned its own bucket) -- a collision among THESE would be a real
    # bug. A gene with no curated symbol (phiX174's K, here) falls back to
    # cosine-nearest-centroid, which is NOT guaranteed collision-free (see
    # assign_target_genes docstring) -- known, documented limitation
    # (dependencies/ortholog_annotation_map.json, gene_symbols_reviewed_no_bucket),
    # not something to hard-fail the self-test over.
    _phrases, gene_symbols, _reviewed = _load_annotation_map()
    curated_genes = [g for g in gene_to_cluster if g in gene_symbols]
    curated_clusters = [gene_to_cluster[g] for g in curated_genes]
    assert len(set(curated_clusters)) == len(curated_clusters), (
        "BUG: two curated-symbol target genes assigned to the same cluster -- "
        "should be impossible by construction of the annotation map"
    )
    fallback_genes = [g for g in gene_to_cluster if g not in gene_symbols]
    if fallback_genes:
        print(f"  NOTE: {fallback_genes} have no curated symbol in the annotation map -- assigned via "
              f"BLAST reciprocal-best-hit bridge, falling back to cosine-nearest-centroid only if "
              f"that finds no single reciprocally-confirmed cluster (see assign_target_genes); "
              f"either path may still collide with another gene's cluster")

    # Stronger discrimination check than diagnostic-2-adjacent's cluster
    # count above: score each real phiX174 gene's own amino-acid sequence
    # against EVERY cluster's table, not just its assigned one. The
    # assigned table should win -- if it doesn't, the table isn't
    # discriminating on anything real (exactly the failure mode this would
    # have caught immediately, instead of the mislabeling passing all
    # existing checks silently).
    # Hard assert for curated-symbol genes (C, D, E, F, G, H, J): their
    # tables are built from real product-text/gene-symbol annotation, so
    # losing this check means the table genuinely isn't discriminating.
    # Fallback genes (K: no curated bucket, BLAST-bridged to its own small,
    # heavily-pooled-backoff cluster -- see the size-distribution printout)
    # only warn: K is phiX174's most obscure, weakly-conserved gene, so
    # losing to a bigger cluster's table is a real, bounded finding about
    # K specifically, not evidence the pipeline is broken.
    for name, aa in gene_seqs.items():
        body = aa[: aa.find("*")] if "*" in aa else aa
        own_cid = gene_to_cluster[name]
        scores = {cid: table.score_aa(body) for cid, table in tables.items()}
        best_cid = max(scores, key=scores.get)
        if best_cid == own_cid:
            continue
        msg = (f"gene {name!r}'s own table ({own_cid}, score={scores[own_cid]:.4f}) does not "
               f"win against {best_cid} (score={scores[best_cid]:.4f}) -- table isn't discriminating")
        if name in gene_symbols:
            raise AssertionError(f"BUG: {msg}")
        print(f"  WARN (fallback gene, known weak signal): {msg}")

    scorer = BlockScorer(tables, gene_to_cluster, PHIX174_GENES)

    # Diagnostic-1-adjacent: the real genome itself must never be gated out.
    for block in blocks:
        real_dna = genome[block.start : block.end]
        real_score = scorer.score_candidate(real_dna, block)
        assert real_score > float("-inf"), (
            f"BUG: real phiX174 sequence was gated out on block {block.genes} "
            f"[{block.start}:{block.end}) -- premature-stop gate or frame math is wrong"
        )
        rng = np.random.default_rng(0)
        shuffled = "".join(rng.permutation(list(real_dna)))
        shuf_score = scorer.score_candidate(shuffled, block)
        print(f"block {block.genes} [{block.start}:{block.end}) len={len(block)}: "
              f"real={real_score:.4f}  shuffled={shuf_score:.4f}  "
              f"{'OK (real > shuffled)' if real_score > shuf_score else 'WARN (real <= shuffled)'}")

    bases = "ACGT"
    rng_codons = [c for c in CODON_TABLE if CODON_TABLE[c] != "*"]

    def fake_generate_fn(prefix: str, n: int, seed: int) -> str:
        local_rng = np.random.default_rng(seed)
        n_codons = (n + 2) // 3
        codons = local_rng.choice(rng_codons, size=n_codons)
        return "".join(codons)[:n]

    block0 = blocks[0]
    prefix = genome[: block0.start]
    cands = generate_block(prefix, block0, c=4, generate_fn=fake_generate_fn)
    assert len(cands) == 4
    assert all(len(cand) == len(block0) for cand in cands)
    scores = [scorer.score_candidate(cand, block0) for cand in cands]
    print(f"\ngenerate_block mock OK: c=4 candidates for block {block0.genes}, "
          f"scores={[round(s, 4) if s != float('-inf') else '-inf' for s in scores]}")


def _validate_target_b(
    genome_fasta: str | Path, cds_fasta: str | Path, ref_fasta: str | Path, prompt_len: int = 0
) -> None:
    """CPU-only, no-GPU validation of prepare_target_b against real target
    genomes: for every accession present in `genome_fasta`, build its
    blocks/gene_to_cluster against the EXISTING reference tables and
    assert the real genome sequence survives BlockScorer's gate on every
    block -- same diagnostic-1-adjacent sanity check `_mock_test` runs for
    phiX174, here run across however many real target genomes are given.
    If prompt_len > 0, also exercises clip_blocks_to_prompt (the real GPU
    driver's prompt almost never lands exactly on a block boundary) --
    scores each CLIPPED block's real tail, not the full block, which is
    the actual thing the driver needs to get right."""
    genomes: Dict[str, str] = {}
    for header, seq in read_fasta(genome_fasta):
        genomes[header.split()[0] if " " in header else header] = seq
    print(f"loaded {len(genomes)} target genomes from {genome_fasta}")

    tables = build_tables([ref_fasta])

    n_ok = 0
    for acc, genome in genomes.items():
        gene_to_cluster, gene_coords, blocks, used_genome = prepare_target_b(acc, genome, cds_fasta, tables)
        n_genes, n_distinct = len(gene_to_cluster), len(set(gene_to_cluster.values()))
        scorer = BlockScorer(tables, gene_to_cluster, gene_coords)
        run_blocks = clip_blocks_to_prompt(blocks, gene_coords, prompt_len) if prompt_len else blocks
        gated = []
        for block in run_blocks:
            real_dna = used_genome[block.start:block.end]
            if len(real_dna) < len(block):
                continue
            if scorer.score_candidate(real_dna, block) == float("-inf"):
                gated.append(block.genes)
        status = "OK" if not gated else f"GATED on {gated}"
        collision = "" if n_distinct == n_genes else f"  collision ({n_distinct}/{n_genes} distinct)"
        clip_note = f" ({len(run_blocks)}/{len(blocks)} blocks survive prompt_len={prompt_len} clipping)" if prompt_len else ""
        print(f"  {acc}: {n_genes} genes, {len(blocks)} blocks{clip_note}{collision} -- real-sequence gate: {status}")
        if status == "OK":
            n_ok += 1
    print(f"\n{n_ok}/{len(genomes)} target genomes: real sequence passes the gate on every block")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mu-sweep", action="store_true",
                     help="leave-one-genome-out likelihood sweep for the backoff strength mu (CPU-only, $0)")
    ap.add_argument("--coverage", metavar="FASTA",
                     help="gene-content coverage check: does a candidate target FASTA (fasta_cds_aa) "
                          "have genes that resolve against the existing 20-genome corpus's tables")
    ap.add_argument("--validate-target-b", nargs=2, metavar=("GENOME_FASTA", "CDS_FASTA"),
                     help="prepare_target_b sanity check: real sequence must pass BlockScorer's gate "
                          "on every block, for every accession in GENOME_FASTA (CPU-only, $0)")
    ap.add_argument("--prompt-len", type=int, default=0,
                     help="with --validate-target-b, also clip blocks to this prompt length and "
                          "verify the clipped tail (not just the full block)")
    a = ap.parse_args()
    if a.validate_target_b:
        default_fasta = Path(__file__).parent / "dependencies" / "microviridae_refs_cds_aa.fasta"
        genome_fasta, cds_fasta = a.validate_target_b
        _validate_target_b(genome_fasta, cds_fasta, default_fasta, prompt_len=a.prompt_len)
    elif a.coverage:
        default_fasta = Path(__file__).parent / "dependencies" / "microviridae_refs_cds_aa.fasta"
        tables = build_tables([default_fasta])
        target_corpus = _load_reference_corpus([a.coverage])
        target_corpus, excluded = _exclude_nested_products(target_corpus)
        if excluded:
            print(f"coverage: excluded {len(excluded)} nested/internal-start product(s) from the target set")
        report = target_coverage(target_corpus, tables)
        print("\ncoverage per target genome (curated / blast / fallthrough / total):")
        for acc, b in sorted(report.per_accession.items()):
            trusted = b["curated"] + b["blast"]
            pct = 100 * trusted / b["total"] if b["total"] else 0.0
            print(f"  {acc:14s} curated={b['curated']:2d} blast={b['blast']:2d} "
                  f"fallthrough={b['fallthrough']:2d} total={b['total']:2d}  trusted={pct:.0f}%")
        t = report.totals
        trusted_total = t["curated"] + t["blast"]
        pct_total = 100 * trusted_total / t["total"] if t["total"] else 0.0
        print(f"\nOVERALL: curated={t['curated']} blast={t['blast']} fallthrough={t['fallthrough']} "
              f"total={t['total']}  trusted={pct_total:.1f}%")
    elif a.mu_sweep:
        default_fasta = Path(__file__).parent / "dependencies" / "microviridae_refs_cds_aa.fasta"
        results = sweep_mu([default_fasta])
        best_mu = max(results, key=results.get)
        print("mu sweep (leave-one-genome-out mean per-residue score, higher is better):")
        for mu in sorted(results):
            marker = "  <-- max" if mu == best_mu else ""
            print(f"  mu={mu:>7.1f}  score={results[mu]:.5f}{marker}")
        print(f"best mu: {best_mu}")
    elif a.mock:
        _mock_test()
    else:
        raise SystemExit("this module has no GPU entrypoint yet -- use --mock or --mu-sweep to test the plumbing")
