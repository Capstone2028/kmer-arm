"""kmer_table.py - DNA k-mer tables and SpecMER-style candidate scoring, CPU only.

Trimmed port of Capstone/kmer-evo2/kmerlib.py's table + scoring half (no generator/
sampler code -- here the "guided generation" loop lives in kmer_fulllength_benchmark.py
and calls EvSpark directly for candidates).

A k-mer is a DNA word k letters long. Count every k-mer in a set of reference genomes
(both strands), turn counts into probabilities -- that's the table. Score a candidate
chunk by looking up each of its k-mers and summing the probabilities (SpecMER, Walton
et al. 2025, Eq. 2, adapted to DNA).

IMPORTANT (avoids circular results): never include the genome you are generating (or
scoring against) in the FASTA files used to build the table.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
_BASES = "ACGT"

_LUT = np.full(256, -1, dtype=np.int64)
for _i, _ch in enumerate(_BASES):
    _LUT[ord(_ch)] = _i
    _LUT[ord(_ch.lower())] = _i


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def read_fasta(path: str | Path) -> Iterator[Tuple[str, str]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    name, parts = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts).upper()
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line)
    if name is not None:
        yield name, "".join(parts).upper()


def encode(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]


def kmer_indices(arr: np.ndarray, k: int) -> np.ndarray:
    n = len(arr) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for j in range(k):
        seg = arr[j : j + n]
        valid &= seg >= 0
        idx = idx * 4 + np.where(seg >= 0, seg, 0)
    return idx[valid]


def count_kmers(seq: str, k: int, both_strands: bool = True) -> np.ndarray:
    counts = np.bincount(kmer_indices(encode(seq), k), minlength=4**k).astype(np.float64)
    if both_strands:
        counts += np.bincount(kmer_indices(encode(revcomp(seq)), k), minlength=4**k)
    return counts


class KmerTable:
    """Counts + probabilities for several k, built from reference genomes."""

    def __init__(self, ks: Sequence[int]):
        self.ks = tuple(sorted(ks))
        self.counts: Dict[int, np.ndarray] = {k: np.zeros(4**k) for k in self.ks}
        self.probs: Dict[int, np.ndarray] = {}
        self.n_letters = 0

    def add_sequence(self, seq: str) -> None:
        self.n_letters += len(seq)
        for k in self.ks:
            self.counts[k] += count_kmers(seq, k, both_strands=True)

    def add_fasta(self, path: str | Path) -> None:
        for _, seq in read_fasta(path):
            self.add_sequence(seq)

    def finalize(self) -> "KmerTable":
        for k in self.ks:
            total = self.counts[k].sum()
            self.probs[k] = (self.counts[k] / total) if total > 0 else self.counts[k]
        return self

    @classmethod
    def from_fastas(cls, ks: Sequence[int], paths: Iterable[str | Path]) -> "KmerTable":
        t = cls(ks)
        for p in paths:
            t.add_fasta(p)
        return t.finalize()

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, ks=np.array(self.ks), **{f"c{k}": self.counts[k] for k in self.ks})

    @classmethod
    def load(cls, path: str | Path) -> "KmerTable":
        z = np.load(path)
        t = cls([int(k) for k in z["ks"]])
        for k in t.ks:
            t.counts[k] = z[f"c{k}"]
        return t.finalize()

    @classmethod
    def from_npz_bytes(cls, data: bytes) -> "KmerTable":
        """Load from raw .npz bytes (for shipping the table into a Modal container
        call without baking it into the image)."""
        import io

        return cls.load(io.BytesIO(data))

    def score(self, seq: str, context: str = "", mode: str = "paper") -> float:
        """Score one candidate chunk.

        mode="paper"    : (1/L) * sum_k sum_i P_k(kmer)  (SpecMER Eq. 2). Small k dominate
                          because P_k is about 1/4^k.
        mode="relative" : same, but each P_k is multiplied by 4^k so every k counts equally.
        mode="loglik"   : (1/L) * sum_k sum_i log(P_k + eps). Stricter: punishes unseen words.

        `context` = sequence already generated (prompt + prior chunks). Its last k-1
        letters are glued in front of the candidate so k-mers straddling the chunk
        boundary are scored too.
        """
        L = len(seq)
        if L == 0:
            return 0.0
        total = 0.0
        for k in self.ks:
            ctx = context[-(k - 1) :] if (k > 1 and context) else ""
            idx = kmer_indices(encode(ctx + seq), k)
            if len(idx) == 0:
                continue
            p = self.probs[k][idx]
            if mode == "paper":
                total += p.sum()
            elif mode == "relative":
                total += (p * 4**k).sum()
            elif mode == "loglik":
                total += np.log(p + 1e-12).sum()
            else:
                raise ValueError(mode)
        return total / L
