"""aa_kmer_table.py - amino-acid k-mer table, CPU only.

Same idea as kmer_table.py's DNA table (count short words in real reference
sequences, turn counts into probabilities), but over translated protein
sequences instead of raw genome DNA, and over the 20-letter amino-acid
alphabet instead of 4-letter DNA. Built from real CDS translations (fetched
from NCBI, not called by a gene caller -- see build_aa_kmer_table.py), so
unlike the DNA table (which has no concept of "coding" at all), this table
is built exclusively from protein-coding sequence.

No reverse complement here -- proteins are read in one direction only, DNA's
two-strands symmetry doesn't apply.

IMPORTANT (avoids circular results): never include phiX174's own proteins in
the FASTA used to build this table -- same circularity rule as kmer_table.py.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
_N_AA = len(AA)

_LUT = np.full(256, -1, dtype=np.int64)
for _i, _ch in enumerate(AA):
    _LUT[ord(_ch)] = _i

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
STOP_CODONS = {"TAA", "TAG", "TGA"}


def translate(dna: str) -> str:
    """Translate DNA (assumed already in-frame, length a multiple of 3 not
    required -- trailing partial codon is dropped) to amino acids. '*' for
    stop, 'X' for any codon containing an N or other non-ACGT letter."""
    dna = dna.upper()
    n = len(dna) - (len(dna) % 3)
    out = []
    for i in range(0, n, 3):
        out.append(CODON_TABLE.get(dna[i : i + 3], "X"))
    return "".join(out)


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
        idx = idx * _N_AA + np.where(seg >= 0, seg, 0)
    return idx[valid]


def count_kmers(seq: str, k: int) -> np.ndarray:
    return np.bincount(kmer_indices(encode(seq), k), minlength=_N_AA**k).astype(np.float64)


class AAKmerTable:
    """Counts + probabilities for several k, built from reference protein sequences."""

    def __init__(self, ks: Sequence[int]):
        self.ks = tuple(sorted(ks))
        self.counts: Dict[int, np.ndarray] = {k: np.zeros(_N_AA**k) for k in self.ks}
        self.probs: Dict[int, np.ndarray] = {}
        self.n_letters = 0

    def add_sequence(self, seq: str) -> None:
        seq = seq.rstrip("*")  # drop trailing stop marker if present (fasta_cds_aa doesn't include it, but be safe)
        self.n_letters += len(seq)
        for k in self.ks:
            self.counts[k] += count_kmers(seq, k)

    def add_fasta(self, path: str | Path) -> None:
        for _, seq in read_fasta(path):
            self.add_sequence(seq)

    def finalize(self) -> "AAKmerTable":
        for k in self.ks:
            total = self.counts[k].sum()
            self.probs[k] = (self.counts[k] / total) if total > 0 else self.counts[k]
        return self

    @classmethod
    def from_fastas(cls, ks: Sequence[int], paths: Iterable[str | Path]) -> "AAKmerTable":
        t = cls(ks)
        for p in paths:
            t.add_fasta(p)
        return t.finalize()

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, ks=np.array(self.ks), **{f"c{k}": self.counts[k] for k in self.ks})

    @classmethod
    def load(cls, path: str | Path) -> "AAKmerTable":
        z = np.load(path)
        t = cls([int(k) for k in z["ks"]])
        for k in t.ks:
            t.counts[k] = z[f"c{k}"]
        return t.finalize()

    @classmethod
    def from_npz_bytes(cls, data: bytes) -> "AAKmerTable":
        import io

        return cls.load(io.BytesIO(data))

    def score_aa(self, aa_seq: str, context: str = "") -> float:
        """mode="relative" only (each P_k * 20**k, equal weight per k) -- the DNA
        arm's default "paper" mode is k=3-dominated by construction (P_k ~ 1/4**k);
        "relative" avoids repeating that here (P_k ~ 1/20**k would be even more
        skewed toward small k on a 20-letter alphabet)."""
        L = len(aa_seq)
        if L == 0:
            return 0.0
        total = 0.0
        for k in self.ks:
            ctx = context[-(k - 1) :] if (k > 1 and context) else ""
            idx = kmer_indices(encode(ctx + aa_seq), k)
            if len(idx) == 0:
                continue
            p = self.probs[k][idx]
            total += (p * _N_AA**k).sum()
        return total / L
