# Phage reference data

`microviridae_reference_records.gb` contains the GenBank records for the 21
accessions in the team's phiX174/Microviridae reference set. It was retrieved
from NCBI Nucleotide with EFetch on 2026-11-11. `phage_stage0_inventory.csv`
summarizes record length, CDS annotations, translation tables, and overlapping
CDS pairs.

The sequences used to build the k-mer table remain in
`../dependencies/microviridae_refs.fasta` and `../phix174.fasta`.

## Recreate the local analysis environment

Use Python 3.11 and install `../requirements-local.txt`.

## Important data check

The records do not all use the same translation table. The set includes
translation tables 1, 4, and 11, so any stop-codon or protein-translation step
must read the table for each record rather than applying one global code.

## Fresh Microviridae collection

The prior Experiment 1 raw folder was not present, so this is a new NCBI
snapshot, not a reconstruction of its split. `microviridae_query_manifest.txt`
records the query and retrieval date; `microviridae_accessions_20261111.txt`
is the exact accession list; `microviridae_records_20261111.gb` is the local
GenBank download (ignored by Git because it is regenerable and about 86 MB).
Use `scripts/fetch_microviridae_genbank.ps1` to fetch it again and
`scripts/summarize_microviridae.py` to regenerate the inventory and summary.

The initial screen returned 7,649 records. Of those, 7,592 have only A/C/G/T
and pass the 4,000–6,500 bp length filter; there are 7,147 distinct exact
sequences after exact-duplicate removal. Of those, 2,549 also have at least
six translated CDS features of 150 nt or longer. This filtered list is in
`microviridae_screened_candidates_20261111.csv`. These are preliminary counts,
not the final experimental split. Exact duplicate removal does not prevent
near-identical genomes from crossing a train/test split; a sequence-similarity
cluster step and the experiment's five-family annotation check are still
needed before selecting calibration and test genomes.

`microviridae_screened_genomes_20261111.fna` contains those 2,549 candidate
genomes. `microviridae_screened_proteins_20261111.faa` contains their
translated proteins, and the matching TSV preserves each protein's accession,
annotation, coordinates, strand, partial-gene flags, and translation table.
The export is reproducible with `sae-arm/scripts/export_candidate_proteins.py`. The
frequent GenBank labels include major capsid, replication initiator, DNA
pilot, and internal scaffolding proteins; these labels help guide the family
search but do not by themselves confirm the five shared families used in the
prior report.

The prepared clustering job is `sae-arm/scripts/run_sequence_clustering_modal.py`.
It uses CPU-only MMseqs2 to make 90% whole-genome similarity groups for the
split leakage check, and 30% protein sequence clusters for exploratory family
mapping. These thresholds are a new operational choice, not a reconstruction
of the team's original clustering definition. After clustering, run
`sae-arm/scripts/summarize_sequence_clusters.py` to build the protein family
summary and genome-by-family matrix under `sae-arm/results/`.
