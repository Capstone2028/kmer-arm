"""build_kmer_table.py - count k-mers in reference genomes and save a table.

IMPORTANT (avoids circular results): never include phiX174 itself in the FASTA files
passed here -- the k-mer table must come from genomes other than the one being
generated/scored.

For the phiX174 experiment, the reference set is dependencies/microviridae_refs.fasta:
20 other Microviridae genomes, ~105.7kb total (~211kb counting both strands).
Chosen for diversity, not just bulk -- a wider k-mer table carries more information
about what "real Microviridae DNA" looks like than a narrow one does, even at small
k, so this mixes two tiers rather than just piling on near-duplicates of phiX174:
  - close relatives: other coliphages in phiX174's own comparative clade (same
    host, same Sinsheimervirus-like group) -- G4 NC_001420.2, alpha3 NC_001330.1,
    ID18 NC_007856.1, ID52 NC_007825.1, WA13 NC_007821.1, ID32 NC_007819.1,
    NC35 NC_007820.1, St-1 NC_012868.1, phiK NC_001730.1.
  - divergent relatives: other Microviridae subfamilies/hosts entirely --
    Bdellovibrio phage phiMH2K NC_002643.1, Chlamydia phages Chp1 NC_001741.1 and
    Chp2 NC_002194.1, Spiroplasma phage 4 NC_003438.1, Parabacteroides phage
    YZ-2015a NC_029012.1, Marine gokushovirus NC_022790.1, environmental
    Gokushovirinae Bog1183_53 NC_027633.1 and Microviridae Fen418_41 NC_027640.1,
    Microvirus CA82 NC_015785.1, and two metagenomic/gut-associated microviruses
    (chimpanzee faeces NC_030472.1, moose faeces NC_040350.1).
Never phiX174 itself (NC_001422.1) -- the circularity rule above.

Default ks=[3, 4, 6]: k=8 (65536 bins) was tried first and dropped even after this
expansion -- at ~211kb of both-strand sequence that's still only ~3.2 counts/bin on
average, too sparse to trust for candidate selection. k=3 is kept (unlike
kmer-evo2's DNA table, which only used k>=4) because it's nearly pure base
composition and this reference set, while now much bigger than the original 4-phage
version, is still small enough that the extra stability is worth it.

Usage:
  python build_kmer_table.py --ks 3 4 6 --fasta dependencies/microviridae_refs.fasta --out dependencies/microviridae_k346.npz
"""
import argparse

from kmer_table import KmerTable

ap = argparse.ArgumentParser()
ap.add_argument("--ks", type=int, nargs="+", required=True)
ap.add_argument("--fasta", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t = KmerTable.from_fastas(a.ks, a.fasta)
t.save(a.out)
print(f"saved {a.out}: ks={t.ks} letters={t.n_letters:,}")
