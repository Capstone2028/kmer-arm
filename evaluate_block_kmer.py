"""Assemble scorable sequences from block_kmer_bench_*.jsonl.

READ THIS BEFORE REPORTING ANY NUMBER FROM THESE FILES.

The 2026-11-14 run's records are NOT a k-mer-guided arm. Two defects,
both verified against block_kmer_bench_ALL.jsonl:

1. Coordinate-frame mismatch. prompts_reliable25_300bp.fa's "__G2"
   records are documented as "the first 300bp of the real genome"
   (prompts_description.txt section 1) but are actually 300bp windows at
   INTERNAL offsets (68..3904; 18 of 25 land exactly on a CDS start).
   block_kmer_benchmark.py uses PROMPT_LEN=300 as an absolute genome
   coordinate (clip_blocks_to_prompt, and the genome[300:...] quality
   window), so BlockScorer gated and PMI-ranked every candidate against
   whichever gene happens to sit at position ~300 -- not the region the
   model was actually continuing. `prompt_offset` below recovers the true
   offset by exact string search, which is what makes any coordinate-
   matched comparison possible at all.

2. The hard gate rejected 100% of candidates (98/98 rounds, every
   genome), so `pick = max(...)` over all-equal -inf silently returned
   index 0 in every round. No selection ever occurred. Independently of
   defect 1, the gate is unsatisfiable at block scale: it requires a
   candidate to carry no stop codon across stop_idx residues AND place
   one exactly at stop_idx (~3e-12 by chance for a 553-residue gene).

What the sequences therefore ARE: genuine, contiguous, UNGUIDED Evo 2
continuations of a real 300bp phage prompt, sampled in block-length
chunks with a per-block seed. There is no fabricated junction --
generate_block conditions each chunk on the entire sequence so far.

So: scoring these is scoring an unguided Evo 2 baseline. Any "k-mer arm
vs plain Evo 2" contrast drawn from them compares unguided to unguided.
Do not report one.

Emits, per record:
  generated.fasta  prompt + block_kmer_seq -- a de novo sequence, safe for
      every reference-free metric (CheckV completeness, core-gene count,
      gene order, start/stop layout, Pfam, PHROG). These need no
      coordinates and are valid as-is.
  natural_matched.fasta  the real genome over the SAME span, from the
      recovered offset, wrapped circularly (these genomes are circular and
      most of these spans cross the origin). For per-position or
      composition comparison only.
  manifest.tsv  offsets, spans, wrap flag, all-gated count.

Run: python evaluate_block_kmer.py [results.jsonl ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
GENOME_FASTA = ROOT / "dependencies" / "prompts28_genomes.fasta"
PROMPTS_FASTA = ROOT / "prompts_reliable25_300bp.fa"
PROMPT_LEN = 300
OUT = ROOT / "eval_block_kmer"


def read_fasta(path):
    name, parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    yield name, "".join(parts)
                name, parts = line[1:], []
            elif line:
                parts.append(line)
    if name:
        yield name, "".join(parts)


def write_fasta(path, records):
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(">" + header + "\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i : i + 70] + "\n")


def revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def prompt_offset(prompt, genome):
    """True start of `prompt` in `genome`, by exact search (the prompt file
    records no coordinate of its own). Searches the circular genome, then
    the reverse complement. Returns (offset, strand), or (None, None) if the
    prompt is not a substring at all -- which would mean the prompt and
    genome files disagree about the sequence itself, not just its framing."""
    circ = genome + genome
    i = circ.find(prompt)
    if 0 <= i < len(genome):
        return i, "+"
    j = revcomp(genome).find(prompt)
    if j >= 0:
        return j, "-"
    return None, None


def main(paths):
    genomes = {h.split()[0]: s for h, s in read_fasta(GENOME_FASTA)}
    prompts = {h.split("__")[0]: s for h, s in read_fasta(PROMPTS_FASTA)}

    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))

    rows, gen_fa, nat_fa, rejected = [], [], [], []
    for r in recs:
        acc = r["accession"]
        seq = r["block_kmer_seq"]
        rounds = r.get("rounds") or []
        genome, prompt = genomes.get(acc), prompts.get(acc)
        if genome is None or prompt is None:
            rejected.append((acc, "missing genome or prompt"))
            continue
        if not rounds:
            rejected.append((acc, "no rounds"))
            continue

        span = sum(b["end"] - b["start"] for b in rounds)
        if span != len(seq):
            rejected.append((acc, "block span %d != len(block_kmer_seq) %d" % (span, len(seq))))
            continue
        if any(rounds[i]["end"] != rounds[i + 1]["start"] for i in range(len(rounds) - 1)):
            rejected.append((acc, "blocks not contiguous -- sequence would be spliced"))
            continue

        off, strand = prompt_offset(prompt, genome)
        if off is None:
            rejected.append((acc, "prompt is not a substring of the genome"))
            continue
        if strand == "-":
            rejected.append((acc, "prompt matches only the reverse complement"))
            continue

        full = prompt + seq
        # Circular: the real continuation of the prompt, same length as what
        # was generated. Most of these spans run past the genome end.
        start = off + PROMPT_LEN
        reps = -(-(start + len(seq)) // len(genome)) + 1
        nat = (genome * reps)[start : start + len(seq)]
        wrapped = start + len(seq) > len(genome)

        n_gated = sum(1 for b in rounds if all(s == float("-inf") for s in b["scores"]))
        seed = r.get("seed")
        gen_fa.append((
            "%s_unguided_s%s prompt_at=%d generated=%d" % (acc, seed, off, len(seq)),
            full,
        ))
        nat_fa.append((
            "%s_natural span=%d..%d circular=%s" % (acc, start, start + len(seq), wrapped),
            nat,
        ))
        rows.append(dict(acc=acc, off=off, nblocks=len(rounds), gen=len(seq),
                         full=len(full), G=len(genome), wrapped=wrapped,
                         gated=n_gated, seed=seed))

    OUT.mkdir(exist_ok=True)
    write_fasta(OUT / "generated.fasta", gen_fa)
    write_fasta(OUT / "natural_matched.fasta", nat_fa)
    with open(OUT / "manifest.tsv", "w") as fh:
        fh.write("accession\tseed\tprompt_offset\tn_blocks\tgenerated_bp\ttotal_bp\t"
                 "genome_bp\tnatural_span_wraps_origin\tall_gated_rounds\n")
        for d in rows:
            fh.write("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%d/%d\n" % (
                d["acc"], d["seed"], d["off"], d["nblocks"], d["gen"],
                d["full"], d["G"], d["wrapped"], d["gated"], d["nblocks"]))

    tot_rounds = sum(d["nblocks"] for d in rows)
    tot_gated = sum(d["gated"] for d in rows)
    print("assembled: %d records" % len(rows))
    print("prompt offsets: min=%d max=%d at_zero=%d" % (
        min(d["off"] for d in rows), max(d["off"] for d in rows),
        sum(1 for d in rows if d["off"] == 0)))
    print("natural span wraps the origin: %d/%d" % (
        sum(1 for d in rows if d["wrapped"]), len(rows)))
    print("rounds with ALL candidates gated: %d/%d (%.0f%%) -- no selection occurred" % (
        tot_gated, tot_rounds, 100.0 * tot_gated / tot_rounds))
    if rejected:
        print("rejected: %d" % len(rejected))
        for acc, why in rejected:
            print("    %s: %s" % (acc, why))
    if tot_gated == tot_rounds:
        print("")
        print("ALL rounds were fully gated: these sequences are an UNGUIDED Evo 2")
        print("baseline, not a k-mer arm. Do not report a guided-vs-plain contrast.")
    print("")
    print("wrote %s/" % OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(ROOT / "block_kmer_bench_ALL.jsonl")])
