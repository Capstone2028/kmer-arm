"""Biological quality scoring for generated sequences, CPU only.

EvSpark's own numbers are all speed (tok/s, speedup, acceptance rate). None of
them say whether the speculative path's *output* is as biologically plausible
as the native path's -- a drafter that gets accepted a lot but nudges the
target distribution would look like a pure win in speculative_benchmark.py's
existing columns while quietly producing worse sequences. This module scores
a sequence (or compares two) so every speed number can be paired with a
quality number.

Cheap tier only, reusing tools already in this repo:
  - coding density / gene calls      -- pyrodigal (as in analyze_generations.py)
  - GC content
  - codon usage divergence           -- frame-0 codon triplets, Jensen-Shannon
  - k-mer divergence                 -- sliding 4-mers, Jensen-Shannon
  - premature-stop proxy             -- min-over-frame forward stop-codon rate
  - CpG observed/expected            -- cpg_o_e, dinucleotide-bias check
  - nullomer fraction                -- nullomer_fraction, missing-5mer coverage
  - homopolymer run stats            -- homopolymer_stats, repeat-collapse check

Pyrodigal is a prokaryotic/phage gene finder, so coding_density/n_genes are
only meaningful for bacterial/viral regions -- pass organism_coding=False
(e.g. for chr21 human_intergenic) to skip them; GC/k-mer/codon/stop metrics
are organism-agnostic and always computed.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pyrodigal

STOP_CODONS = {"TAA", "TAG", "TGA"}
BASES = "ACGT"

FINDER_CLS = getattr(pyrodigal, "GeneFinder", None) or pyrodigal.OrfFinder
_FINDER = FINDER_CLS(meta=True)  # meta mode: needed for short/non-training sequences


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return round((seq.count("G") + seq.count("C")) / max(len(seq), 1), 4)


def ngram_freqs(seq: str, n: int, step: int = 1) -> dict[str, float]:
    """Normalized frequency of every length-``n`` substring, stepping by ``step``
    (step=1 for sliding k-mers, step=3/frame-aligned for codon usage). Windows
    containing anything outside ACGT are skipped."""
    seq = seq.upper()
    counts = Counter()
    total = 0
    for i in range(0, len(seq) - n + 1, step):
        gram = seq[i : i + n]
        if any(c not in BASES for c in gram):
            continue
        counts[gram] += 1
        total += 1
    if total == 0:
        return {}
    return {g: c / total for g, c in counts.items()}


def _js_divergence(freqs_a: dict[str, float], freqs_b: dict[str, float]) -> float | None:
    """Jensen-Shannon divergence (bits, base-2) between two sparse frequency
    dicts. Returns None if either side has no usable grams."""
    if not freqs_a or not freqs_b:
        return None
    keys = sorted(set(freqs_a) | set(freqs_b))
    p = np.array([freqs_a.get(k, 0.0) for k in keys])
    q = np.array([freqs_b.get(k, 0.0) for k in keys])
    m = (p + q) / 2.0

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return round(0.5 * kl(p, m) + 0.5 * kl(q, m), 5)


def kmer_js_divergence(seq_a: str, seq_b: str, k: int = 4) -> float | None:
    return _js_divergence(ngram_freqs(seq_a, k, step=1), ngram_freqs(seq_b, k, step=1))


def codon_js_divergence(seq_a: str, seq_b: str, frame: int = 0) -> float | None:
    """Codon-usage divergence, both sequences read from the same frame offset.
    Valid as a direct comparison only when seq_a/seq_b start at the same
    genomic position (true here: native/speculative/real are all continuations
    of the identical prompt, so frame 0 for all three means the same frame)."""
    return _js_divergence(
        ngram_freqs(seq_a[frame:], 3, step=3), ngram_freqs(seq_b[frame:], 3, step=3)
    )


def frame_stop_rates(seq: str) -> dict[str, float]:
    """Forward-strand stop-codon rate (stops per 100 codons) in each of the 3
    reading frames. The frame actually carrying a gene should sit well below
    the other two and below the ~4.7% (3/64) codon-level baseline; a model
    that's losing codon structure shows elevated rate even in its best frame.
    ``min_frame`` is the premature-stop proxy: the best-case reading frame's
    stop rate, since we don't carry true annotated frame through generation."""
    seq = seq.upper()
    rates = {}
    for frame in range(3):
        freqs = ngram_freqs(seq, 3, step=3) if frame == 0 else ngram_freqs(seq[frame:], 3, step=3)
        total = sum(freqs.values())  # always ~1.0 if any codons counted, else 0
        stop_freq = sum(freqs.get(c, 0.0) for c in STOP_CODONS)
        rates[f"frame{frame}"] = round(stop_freq * 100, 3) if total else None
    valid = [v for v in rates.values() if v is not None]
    rates["min_frame"] = min(valid) if valid else None
    return rates


def kmer_entropy(seq: str, k: int = 4) -> float | None:
    """Shannon entropy of the sliding k-mer distribution, normalized by the
    max possible entropy (log2(4**k)) to land in [0, 1]. Catches repeat
    collapse -- a model degenerating into e.g. "ACACACAC..." -- which none of
    the other metrics here are guaranteed to flag (a short homopolymer/repeat
    run can still land on plausible GC and even nonzero coding density by
    chance). Low score = repetitive/degenerate; ~1.0 = diverse, real-DNA-like."""
    freqs = ngram_freqs(seq, k, step=1)
    if not freqs:
        return None
    probs = np.array(list(freqs.values()))
    h = -float(np.sum(probs * np.log2(probs)))
    h_max = 2 * k  # log2(4**k)
    return round(h / h_max, 4)


def cpg_o_e(seq: str) -> float | None:
    """CpG observed/expected ratio (Gardiner-Garden & Frommer 1987): observed
    CpG dinucleotide count divided by the count expected from the sequence's
    own C and G mononucleotide frequencies alone, i.e. what you'd see if C and
    G occurred independently at their observed rates. Real genomic DNA is
    typically CpG-depleted (O/E << 1, methylation-driven in vertebrates; also
    suppressed in many phage/bacterial genomes via other mutational biases) --
    a generated sequence that's lost this bias would read closer to 1.0."""
    seq = seq.upper()
    n = len(seq)
    if n < 2:
        return None
    observed = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i + 1] == "G")
    total_dinuc = n - 1
    p_c, p_g = seq.count("C") / n, seq.count("G") / n
    expected = p_c * p_g * total_dinuc
    return round(observed / expected, 4) if expected > 0 else None


def nullomer_fraction(seq: str, k: int = 5) -> float | None:
    """Fraction of all 4**k possible length-k words that never appear in this
    sequence (position-invariant -- counts distinct observed k-mers, not
    where they sit). A model that's collapsed onto a narrower sub-vocabulary
    of k-mers than real DNA actually uses will show a higher nullomer
    fraction than a same-length real/native comparison at the same k."""
    freqs = ngram_freqs(seq, k, step=1)
    if not freqs and len(seq) < k:
        return None
    possible = 4 ** k
    return round(1 - len(freqs) / possible, 4)


def homopolymer_stats(seq: str, min_run: int = 4) -> dict:
    """homopolymer_fraction: share of the sequence covered by same-base runs
    of length >= min_run (real DNA has some but not much of this; excessive
    repeat-collapse is a known failure mode for generative models, same
    concern bio_scoring's kmer_entropy already targets at the 4-mer level --
    this is the single-base-run-length view of the same thing).
    longest_homopolymer: the single longest same-base run, bp."""
    seq = seq.upper()
    if not seq:
        return {"homopolymer_fraction": None, "longest_homopolymer": 0}
    runs = []
    cur_char, cur_len = seq[0], 1
    for c in seq[1:]:
        if c == cur_char:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_char, cur_len = c, 1
    runs.append(cur_len)
    covered = sum(r for r in runs if r >= min_run)
    return {
        "homopolymer_fraction": round(covered / len(seq), 4),
        "longest_homopolymer": max(runs),
    }


def pyrodigal_stats(seq: str) -> dict:
    genes = _FINDER.find_genes(seq.encode())
    spans = sorted((min(g.begin, g.end), max(g.begin, g.end)) for g in genes)
    covered, cur_lo, cur_hi = 0, None, None
    for lo, hi in spans:
        if cur_hi is None or lo > cur_hi + 1:
            if cur_hi is not None:
                covered += cur_hi - cur_lo + 1
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        covered += cur_hi - cur_lo + 1
    complete = [g for g in genes if not getattr(g, "partial_begin", False) and not getattr(g, "partial_end", False)]
    return {
        "coding_density": round(min(covered, len(seq)) / max(len(seq), 1), 4),
        "n_genes": len(genes),
        "n_complete_genes": len(complete),
        "longest_gene_bp": max((abs(g.end - g.begin) + 1 for g in genes), default=0),
    }


def score_sequence(seq: str, organism_coding: bool = True) -> dict:
    """Metrics for a single sequence in isolation (no comparison to a reference)."""
    out = {
        "length": len(seq), "gc": gc_content(seq),
        "complexity": kmer_entropy(seq, k=4), **frame_stop_rates(seq),
        "cpg_o_e": cpg_o_e(seq),
        "nullomer_fraction_5mer": nullomer_fraction(seq, k=5),
        **homopolymer_stats(seq, min_run=4),
    }
    if organism_coding:
        out.update(pyrodigal_stats(seq))
    return out


def compare_sequences(seq_a: str, seq_b: str, organism_coding: bool = True) -> dict:
    """Pairwise comparison, e.g. native vs. speculative or native vs. real
    continuation. Includes each side's own score_sequence plus divergence
    metrics between them."""
    stats_a = score_sequence(seq_a, organism_coding)
    stats_b = score_sequence(seq_b, organism_coding)
    diff = {}
    for key in ("gc", "coding_density", "n_genes", "longest_gene_bp"):
        va, vb = stats_a.get(key), stats_b.get(key)
        if va is not None and vb is not None:
            diff[f"{key}_diff"] = round(va - vb, 4)
    return {
        "a": stats_a,
        "b": stats_b,
        "kmer_js_divergence": kmer_js_divergence(seq_a, seq_b, k=4),
        "codon_js_divergence": codon_js_divergence(seq_a, seq_b, frame=0),
        **diff,
    }
