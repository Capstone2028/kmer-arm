"""build_aa_kmer_table.py - count amino-acid k-mers in reference CDS translations.

IMPORTANT (avoids circular results): never include phiX174's own proteins in the
FASTA passed here -- same circularity rule as build_kmer_table.py.

Reference set: dependencies/microviridae_refs_cds_aa.fasta -- real CDS
translations (fasta_cds_aa) fetched from NCBI efetch for the same 20 other
Microviridae/relative genomes used for the DNA table (see build_kmer_table.py's
docstring for the full accession list and why those 20). Pulling real annotated
translations rather than running a gene caller on the reference side avoids
reinjecting gene-caller blind spots into the very table meant to go around them.

168 CDS, ~34.8k amino acids total. At the 20-letter amino-acid alphabet, k=4
(160,000 bins) would be far too sparse off this corpus; k=2 (400 bins, ~87
counts/bin) and k=3 (8000 bins, ~4.3 counts/bin) are the only ones with enough
density to trust -- see BUILD_NOTES.md.

Usage:
  python build_aa_kmer_table.py --ks 2 3 --fasta dependencies/microviridae_refs_cds_aa.fasta --out dependencies/microviridae_aa_k23.npz
"""
import argparse

from aa_kmer_table import AAKmerTable

ap = argparse.ArgumentParser()
ap.add_argument("--ks", type=int, nargs="+", required=True)
ap.add_argument("--fasta", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t = AAKmerTable.from_fastas(a.ks, a.fasta)
t.save(a.out)
print(f"saved {a.out}: ks={t.ks} letters={t.n_letters:,}")
