"""Whole-replicon biological scoring for phiX174 full-length generations, CPU only.

bio_scoring.py's metrics (GC, k-mer/codon divergence, Pyrodigal coding density)
were built for ~1kb windows and are organism-agnostic. This module adds the
genome-scale checks that only make sense once a generation spans the whole
~5.4kb replicon, and that are specific to phiX174's own real annotation:

  - orf_closure        -- does each of phiX174's 11 real genes survive intact
                           (start codon, in-frame, no premature stop, correct
                           stop) at its real coordinates in the generated
                           sequence? Uses ground-truth coordinates from NCBI
                           (NC_001422.1 / GCF_000819615.1), not a gene caller --
                           sidesteps the problem the Arc Institute phage-design
                           paper (biorxiv 2025.09.12.675911) hit, where
                           standard annotators (Prodigal included) undercount
                           phiX174's overlapping ORFs. This only works because
                           prompt+generation is a linear continuation of the
                           same coordinate frame as the real genome -- it is
                           NOT a general-purpose caller and should not be
                           reused for any other organism.
  - gc_skew             -- cumulative (G-C) walk and its zero-crossings.
                           Real bacterial/phage replicons show a small number
                           of sign changes tied to the origin/terminus; a
                           generated sequence that's lost global compositional
                           structure will show a noisier, more frequent-crossing
                           curve.
  - terminator_scan     -- heuristic rho-independent terminator proxy: a
                           GC-rich inverted-repeat (hairpin stem) followed
                           within a short spacer by a run of T's (the DNA-
                           strand analog of the poly-U tail in the nascent
                           transcript). This is NOT a validated terminator
                           caller (the Arc paper didn't formalize one either --
                           they only analyzed terminators qualitatively) --
                           it is a cheap structural proxy, reported as a count/
                           position list, not a pass/fail score.

Gene coordinates below are 0-based half-open, converted from check_annotation.py's
1-based inclusive output (--gbff ncbi_dataset_extracted/ncbi_dataset/data/
GCF_000819615.1/genomic.gbff) via start-1 / end unchanged. Locus tags per
NCBI, not classic phage gene letters -- phiX174p01 = gene A, p02 = gene A*,
p03 = gene B... mapped below for readability since the classic letters are
what the literature (incl. the Arc paper) uses.
"""
from __future__ import annotations

import numpy as np

START_CODONS = {"ATG", "GTG", "TTG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

GENOME_LEN = 5386  # NC_001422.1

# name, 0-based segments in document order (join() order for origin-spanning
# genes), strand. All +1 strand on this genome.
PHIX174_GENES = [
    {"name": "A",  "segments": [(3980, 5386), (0, 136)],  "wraps_origin": True},
    {"name": "A*", "segments": [(4496, 5386), (0, 136)],  "wraps_origin": True},
    {"name": "B",  "segments": [(5074, 5386), (0, 51)],   "wraps_origin": True},
    {"name": "C",  "segments": [(50, 221)],                "wraps_origin": False},
    {"name": "D",  "segments": [(132, 393)],                "wraps_origin": False},
    {"name": "F",  "segments": [(389, 848)],                "wraps_origin": False},
    {"name": "G",  "segments": [(567, 843)],                "wraps_origin": False},
    {"name": "H",  "segments": [(847, 964)],                "wraps_origin": False},
    {"name": "E",  "segments": [(1000, 2284)],              "wraps_origin": False},
    {"name": "J",  "segments": [(2394, 2922)],              "wraps_origin": False},
    {"name": "K",  "segments": [(2930, 3917)],              "wraps_origin": False},
]


def _extract(seq: str, segments: list[tuple[int, int]]) -> str:
    return "".join(seq[lo:hi] for lo, hi in segments)


def orf_closure_check(seq: str, prompt_len: int = 0, genes: list[dict] | None = None) -> dict:
    """Checks each real phiX174 gene against the generated sequence at its
    real coordinates. Genes that wrap the circular origin are skipped unless
    the full generation covers position 0 through the wrap point *and* that
    region came from this same linear sequence (i.e. prompt_len == 0) --
    otherwise the origin-spanning segment doesn't exist in a one-shot linear
    continuation and would always read as broken for a reason that has
    nothing to do with generation quality. Genes entirely before prompt_len
    are also skipped (that's prompt, not generated content, nothing to score).
    """
    genes = genes if genes is not None else PHIX174_GENES
    results = []
    for g in genes:
        segments = g["segments"]
        gene_start = min(lo for lo, _ in segments)
        gene_end = max(hi for _, hi in segments)
        if g["wraps_origin"] and prompt_len > 0:
            results.append({"name": g["name"], "skipped": "wraps_origin_not_covered"})
            continue
        if gene_end <= prompt_len:
            results.append({"name": g["name"], "skipped": "entirely_in_prompt"})
            continue
        if gene_start < prompt_len:
            results.append({"name": g["name"], "skipped": "partially_in_prompt"})
            continue
        if gene_end > len(seq):
            results.append({"name": g["name"], "skipped": "beyond_generated_length"})
            continue

        sub = _extract(seq, segments).upper()
        length = len(sub)
        start_codon = sub[:3]
        stop_codon = sub[-3:]
        codons = [sub[i:i + 3] for i in range(0, length - 2, 3)]
        internal_stops = sum(1 for c in codons[:-1] if c in STOP_CODONS)

        ok_len = length % 3 == 0
        ok_start = start_codon in START_CODONS
        ok_stop = stop_codon in STOP_CODONS
        ok_no_premature_stop = internal_stops == 0
        closed = ok_len and ok_start and ok_stop and ok_no_premature_stop

        results.append({
            "name": g["name"], "closed": closed, "length": length,
            "start_codon": start_codon, "stop_codon": stop_codon,
            "ok_len": ok_len, "ok_start": ok_start, "ok_stop": ok_stop,
            "internal_premature_stops": internal_stops,
        })

    scored = [r for r in results if "closed" in r]
    n_closed = sum(1 for r in scored if r["closed"])
    return {
        "genes": results,
        "n_scored": len(scored),
        "n_closed": n_closed,
        "closure_fraction": round(n_closed / len(scored), 4) if scored else None,
    }


def gc_skew_sign_changes(seq: str, smooth_window: int = 51) -> dict:
    """Cumulative (G-C) walk, smoothed, then zero-crossing positions. Returns
    the raw crossing count and positions -- interpretation (e.g. "does this
    land near the known origin at position 0") is left to the caller, since
    a linear generated sequence isn't circular and the origin-region framing
    only applies cleanly when prompt_len==0."""
    s = seq.upper()
    step = np.where(np.array(list(s)) == "G", 1, np.where(np.array(list(s)) == "C", -1, 0))
    cum = np.cumsum(step)
    if smooth_window > 1 and len(cum) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        curve = np.convolve(cum, kernel, mode="valid")
    else:
        curve = cum.astype(float)
    signs = np.sign(curve)
    signs[signs == 0] = 1  # treat exact zero as continuation of previous sign, avoid spurious crossings
    crossings = np.where(np.diff(signs) != 0)[0].tolist()
    return {
        "n_sign_changes": len(crossings),
        "sign_change_positions": crossings,
        "final_cumulative_skew": int(cum[-1]) if len(cum) else 0,
    }


def _is_near_complement(a: str, b_reversed: str, max_mismatch: int) -> bool:
    mismatches = 0
    for x, y in zip(a, b_reversed):
        if x != COMPLEMENT.get(y, "?"):
            mismatches += 1
            if mismatches > max_mismatch:
                return False
    return True


def terminator_scan(seq: str, stem_range=(4, 9), loop_range=(3, 8), spacer_max: int = 3,
                     min_t_run: int = 4, max_mismatch: int = 1) -> dict:
    """Heuristic rho-independent terminator proxy -- see module docstring for
    caveats. Scans every position for a GC-rich hairpin (stem within
    ``stem_range`` bp, loop within ``loop_range`` bp, up to ``max_mismatch``
    stem mismatches) followed within ``spacer_max`` bp by a run of at least
    ``min_t_run`` T's. Overlapping hits at the same approximate site are
    collapsed to the longest stem found there."""
    s = seq.upper()
    n = len(s)
    hits_by_pos = {}
    for i in range(n):
        for stem in range(stem_range[0], stem_range[1] + 1):
            left = s[i:i + stem]
            if len(left) < stem or "N" in left:
                continue
            for loop in range(loop_range[0], loop_range[1] + 1):
                j = i + stem + loop
                k = j + stem
                if k > n:
                    continue
                right = s[j:k]
                if not _is_near_complement(left, right[::-1], max_mismatch):
                    continue
                gc = sum(1 for c in left if c in "GC")
                if gc < stem * 0.5:  # require a GC-rich stem, not just any near-palindrome
                    continue
                tail = s[k:k + spacer_max + min_t_run]
                if "T" * min_t_run in tail:
                    prev = hits_by_pos.get(i)
                    if prev is None or stem > prev["stem_len"]:
                        hits_by_pos[i] = {"pos": i, "stem_len": stem, "loop_len": loop}
                    break  # found a hit for this (i, stem); don't also try longer loops
    hits = sorted(hits_by_pos.values(), key=lambda h: h["pos"])
    return {"n_hits": len(hits), "hits": hits}


def score_genome(seq: str, prompt_len: int = 0, genes: list[dict] | None = None) -> dict:
    """Whole-replicon metrics for one full-length (or near-full-length)
    phiX174 generation. Complements, not replaces, bio_scoring.score_sequence
    (GC/k-mer/codon/coding_density still apply and should be computed too)."""
    return {
        "length": len(seq),
        "orf_closure": orf_closure_check(seq, prompt_len, genes),
        "gc_skew": gc_skew_sign_changes(seq),
        "terminators": terminator_scan(seq),
    }
