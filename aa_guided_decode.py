"""aa_guided_decode.py - per-codon amino-acid k-mer guidance for token-level generation.

Not best-of-c chunk scoring (kmer_fulllength_benchmark.py's score_mode="aa_gated") --
this reweights the model's OWN per-token sampling distribution at the exact moment
each codon closes (its 3rd nucleotide), instead of generating whole chunks and
picking among complete candidates after the fact. See BUILD_NOTES.md for why no
lookahead is needed: bases 1-2 of a codon are unguided (no complete amino-acid
word exists yet to score); only base 3, which fully determines one amino acid,
gets gated (premature stop -> forbidden) and reweighted (by the amino-acid k-mer
table's score for that resulting amino acid given the gene's own amino acids so
far). Pure, GPU-free logic -- the actual token loop (which needs EvSpark's raw
model/logits access) lives in kmer_guided_fulllength_benchmark.py and duplicates
these functions inline (same reason kmer_fulllength_benchmark.py duplicates
kmer_table.py: the Modal container doesn't have local repo files baked in).

Design decisions baked in here (see BUILD_NOTES.md for the reasoning):
  - Forces the gene's real terminal stop codon exactly at its real end position
    (pos == e-1) -- any non-stop candidate there is forbidden, not just allowed
    to compete. Directly targets orf_closure_check's exact-coordinate test.
  - Overlapping genes closing at the same position (different frames,
    coinciding phase): forbidden if EITHER gene's rule forbids; continuous
    weights are multiplied together (not averaged) -- a base must look good
    under BOTH frames simultaneously, matching the biological requirement that
    both overlapping proteins have to actually work there.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from aa_kmer_table import AA, AAKmerTable, CODON_TABLE

BASES = "ACGT"

# phiX174 genes, 0-based half-open, non-origin-wrapping only (A/A*/B excluded --
# see kmer_fulllength_benchmark.py's PHIX174_GENES_NOWRAP for the same rationale).
PHIX174_GENES_NOWRAP = [
    (50, 221),    # C
    (132, 393),   # D
    (389, 848),   # F
    (567, 843),   # G
    (847, 964),   # H
    (1000, 2284), # E
    (2394, 2922), # J
    (2930, 3917), # K
]

_AA_IDX = {ch: i for i, ch in enumerate(AA)}


def translate_codons(dna: str) -> str:
    dna = dna.upper()
    n = len(dna) - (len(dna) % 3)
    return "".join(CODON_TABLE.get(dna[i : i + 3], "X") for i in range(0, n, 3))


def seed_gene_aa(prompt: str, prompt_len: int) -> Dict[int, str]:
    """Initial amino-acid context per gene from whatever complete codons of
    that gene already sit in the prompt (empty string if the gene starts
    after the prompt)."""
    gene_aa: Dict[int, str] = {}
    for gi, (s, e) in enumerate(PHIX174_GENES_NOWRAP):
        if s < prompt_len:
            codon_end = s + 3 * ((min(e, prompt_len) - s) // 3)
            gene_aa[gi] = translate_codons(prompt[s:codon_end])
        else:
            gene_aa[gi] = ""
    return gene_aa


def closing_genes(pos: int) -> List[Tuple[int, int, int]]:
    """Which genes (gene_index, start, end) have their codon closing AT this
    absolute genome position (i.e. `pos` is the 3rd base of one of their
    codons)."""
    return [
        (gi, s, e)
        for gi, (s, e) in enumerate(PHIX174_GENES_NOWRAP)
        if s <= pos < e and (pos - s) % 3 == 2
    ]


def aa_transition_weight(context: str, candidate: str, table: AAKmerTable) -> float:
    """Mean "relative"-mode score (P_k * 20**k, equal weight per k) for the
    k-mer window(s) ending in `candidate`, given the amino acids already
    placed (`context`). Mirrors AAKmerTable.score_aa's per-k weighting but for
    a single new letter instead of a whole sequence."""
    total = 0.0
    n = 0
    for k in table.ks:
        if len(context) < k - 1:
            continue
        window = (context[-(k - 1) :] if k > 1 else "") + candidate
        idx = 0
        for ch in window:
            idx = idx * 20 + _AA_IDX[ch]
        total += table.probs[k][idx] * 20**k
        n += 1
    return total / n if n > 0 else 1e-6


def candidate_weights(
    seq: str, pos: int, closing: List[Tuple[int, int, int]], gene_aa: Dict[int, str], table: AAKmerTable
) -> Dict[str, Tuple[bool, float]]:
    """For each of the 4 bases, whether it's allowed at this closing position
    and its combined continuous weight across all active-overlap genes.
    `seq` is the sequence generated so far (positions < pos known; pos itself
    not yet decided)."""
    result: Dict[str, Tuple[bool, float]] = {b: (True, 1.0) for b in BASES}
    for gi, s, e in closing:
        is_final = pos == e - 1
        for b in BASES:
            allowed, weight = result[b]
            if not allowed:
                continue
            codon = seq[pos - 2] + seq[pos - 1] + b
            aa = CODON_TABLE.get(codon, "X")
            if is_final:
                if aa != "*":
                    result[b] = (False, weight)
                # correct terminal stop: allowed, weight unchanged (neutral)
            else:
                if aa == "*" or aa == "X":
                    result[b] = (False, weight)
                else:
                    w = aa_transition_weight(gene_aa[gi], aa, table)
                    result[b] = (True, weight * w)
    return result


def advance_gene_aa(seq_with_new_base: str, pos: int, closing: List[Tuple[int, int, int]], gene_aa: Dict[int, str]) -> None:
    """After the base at `pos` has actually been sampled, append the
    resulting amino acid to each closing gene's running context (mutates
    gene_aa in place). Skips stop codons (table has no '*' entries)."""
    for gi, s, e in closing:
        codon = seq_with_new_base[pos - 2 : pos + 1]
        aa = CODON_TABLE.get(codon, "X")
        if aa not in ("*", "X"):
            gene_aa[gi] = gene_aa[gi] + aa
