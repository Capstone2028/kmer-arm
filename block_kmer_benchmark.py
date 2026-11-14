"""Experiment 6: block-pipeline k-mer-guided generation, Target B (Microviridae
family members other than phiX174).

Builds on block_pipeline.py's annotation-first + BLAST reference-corpus
clustering (CPU-only, built ONCE locally, before any GPU call -- see
BUILD_NOTES.md 2026-11-13) and prepare_target_b (the generic per-target
wrapper, validated this session via block_pipeline.py --validate-target-b:
25/25 real target sequences pass BlockScorer's gate on every block at
prompt_len=300, the same real sequences used here).

Mechanics: per target genome, grow the sequence one real-gene BLOCK at a
time (block_pipeline.make_blocks/clip_blocks_to_prompt -- NOT a fixed
chunk_len the way kmer_fulllength_benchmark.py's phiX174 arm works). Each
block, make --c EvSpark generate_native() calls with distinct seeds
(block_pipeline.generate_block), score every candidate with BlockScorer
(translate each active gene in its own real frame, hard-gate on a
premature/missing terminal stop, rank gate-survivors by mean PMI against
the amino-acid k-mer cluster tables), keep the highest-scoring candidate,
repeat for the next block.

Cost note, same shape as kmer_fulllength_benchmark.py's: --c candidates
per block means --c EvSpark generate_native() calls, not one, and
EvSpark's generate_native is hard-coded to batch=1 (EvSpark/evspark/
api.py) -- "c candidates nearly free" does NOT hold until EvSpark gains
real batching. Default c=3 for exactly that reason; don't read a larger
--c as a free knob.

Prompts: prompts_reliable25_300bp.fa (25 genomes, real 300bp prompt each
-- the 28-genome coverage-filtered set minus 3 more that are entirely
minus-strand; see prompts_description.txt section 6 for why those 3 can't
use this script's incremental block-by-block generation at all, not a
data-quality problem). The native-only baseline for these prompts is
handled elsewhere, by another part of the team -- this script is the
k-mer arm only.

Run:  BENCH_GPU=L40S GPU_PRICE_PER_HR=1.95 modal run block_kmer_benchmark.py
Options: --c 3 --seeds 0 --only <accession-prefix-or-comma-list> --tag <name>
Mock-test the plumbing (no GPU, $0): python block_kmer_benchmark.py --mock
Writes: block_kmer_bench_<GPU>[_<tag>].jsonl -- one line appended per
        (accession, seed) the MOMENT it finishes, from the local entrypoint
        (this machine, not the remote GPU container, so a crash in a LATER
        genome can't erase earlier ones). Resumable: rerunning the same
        --tag skips any (accession, seed) already in that file. A failed
        genome (OOM or otherwise) is caught, logged to
        block_kmer_failures_<GPU>[_<tag>].jsonl, and does not stop the rest
        of the run. Genomes are processed longest-first, so a GPU-headroom
        problem shows up in the first call, not forty minutes in.

2026-11-13 CUDA OOM, found on a real run: prefill processes the whole
growing context in one parallel FFT pass, so activation memory scales with
context length and spikes hard at ~4-5kb (confirmed NOT a cross-call leak --
re-running the exact same genome/block twice gave IDENTICAL "tried to
allocate"/"allocated by PyTorch" numbers). Image sets
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to let that one large
allocation come from fragmented memory instead of requiring one contiguous
block -- the fix PyTorch's own error message names for exactly this
symptom. If a genome still OOMs with that set, it's logged to the failures
file and the run continues past it rather than stopping.
"""
import json
import time
from pathlib import Path

import modal

from speculative_benchmark import GPU, PRICE, app, read_fasta_records, image, EVSPARK_DIR

# See module docstring's 2026-11-13 OOM note. Must be set before CUDA
# initializes, so on the image's env (container-wide), not inside the function.
# Must come BEFORE add_local_python_source below -- Modal requires
# add_local_* to be the LAST step in an image chain (hit this directly:
# "An image tried to run a build step after using image.add_local_*").
image = image.env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
# Same reason kmer_fulllength_benchmark.py does this for speculative_benchmark:
# the shared `image` above never bakes in sibling repo files on its own, so
# without this the remote container crash-loops on ModuleNotFoundError. Real
# module imports here (not the manual npz-byte-serialization the OLDER scripts
# used) -- block_pipeline.py's annotation-first + BLAST clustering is too much
# logic to duplicate inline and keep in sync by hand.
image = image.add_local_python_source("speculative_benchmark", "block_pipeline", "aa_kmer_table", "kmer_table")

ROOT = Path(__file__).parent
REF_FASTA = ROOT / "dependencies" / "microviridae_refs_cds_aa.fasta"
CDS_FASTA = ROOT / "dependencies" / "prompts40_cds_aa.fasta"
GENOME_FASTA = ROOT / "dependencies" / "prompts28_genomes.fasta"
PROMPTS_FASTA = ROOT / "prompts_reliable25_300bp.fa"
# The LENGTH of each prompt, and nothing more. It is NOT a genome coordinate:
# see _prompt_offset and the module docstring's 2026-11-14 note.
PROMPT_LEN = 300


def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _prompt_offset(prompt: str, genome: str) -> int | None:
    """Where `prompt` actually starts in `genome`, by exact substring search
    over the circular sequence. Returns None if it is not present at all
    (prompt file and genome file disagreeing on the sequence itself, which is
    a different and worse problem than disagreeing on the framing).

    This exists because prompts_description.txt section 1 says the "__G2"
    prompts are "the first 300bp of the real genome" and they are not -- they
    are 300bp windows at internal offsets 68..3904, 18 of 25 landing exactly
    on a CDS start. The prompt files carry no coordinate of their own, so the
    offset has to be recovered. Verified 2026-11-14: 0 of 25 start at 0.
    """
    i = (genome + genome).find(prompt)
    if 0 <= i < len(genome):
        return i
    return None if _revcomp(genome).find(prompt) < 0 else None


def _circular_slice(genome: str, start: int, n: int) -> str:
    """`n` bases of `genome` from `start`, wrapping the origin. These genomes
    are circular and, measured on the 2026-11-14 run, 21 of 23 generated
    spans run past the genome end -- a plain genome[start:start+n] silently
    returns a SHORT string for those, which then skips the length-matched
    comparison below without saying so."""
    if n <= 0:
        return ""
    reps = -(-(start + n) // len(genome)) + 1
    return (genome * reps)[start : start + n]


@app.function(image=image, gpu=GPU, timeout=3 * 60 * 60)
def run_block_kmer_prompt(
    model_name: str, ckpt: str, name: str, accession: str, prompt: str,
    gene_to_cluster: dict, gene_coords: dict, blocks: list, tables: dict,
    c: int, seed: int, gpu: str, price: float,
) -> dict:
    import torch
    from evspark import EvSpark
    from evspark.api import clean_prompt
    from block_pipeline import BlockScorer, generate_block

    scorer = BlockScorer(tables, gene_to_cluster, gene_coords)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    es = EvSpark.load(ckpt=ckpt, model_name=model_name, device="cuda:0")
    load_s = time.perf_counter() - t0

    prompt = clean_prompt(prompt)

    # --- warm-up, untimed (see speculative_benchmark.run_prompt for rationale) ---
    WARMUP_TOKENS = 32
    es.generate_native(prompt, n_tokens=WARMUP_TOKENS, greedy=False, seed=seed,
                        temperature=1.0, top_k=4)
    torch.cuda.synchronize()

    def generate_fn(prefix: str, n: int, s: int) -> str:
        text = es.generate_native(prefix, n_tokens=n, greedy=False, seed=s,
                                   temperature=1.0, top_k=4).text
        # NOT a cross-call leak -- confirmed 2026-11-13 by re-running the
        # exact same genome/block twice (one run with empty_cache() added
        # here, one without) and getting IDENTICAL numbers both times
        # ("Tried to allocate 15.82 GiB" / "26.27 GiB allocated by PyTorch").
        # empty_cache() changed nothing, which rules out "stale objects from
        # earlier calls aren't being freed". What's actually happening:
        # prefill processes the WHOLE growing context in one parallel FFT
        # pass (vortex's prefill_via_modal_fft), so activation memory scales
        # with context length and spikes far above steady-state decode
        # memory at ~4-5kb contexts -- a real peak, not a leak. The old
        # chunk_len=512 script re-prefills the same growing context too, so
        # it hits a SIMILAR peak, just never in a single call large enough
        # to need one contiguous 15+GB block -- which is what
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set on the image
        # below) actually fixes: serving that request from fragmented
        # non-contiguous memory instead of requiring one unbroken region.
        # empty_cache() stays here anyway -- still cheap, still correct
        # practice, just not the fix.
        torch.cuda.empty_cache()
        return text

    seq = prompt
    rounds = []
    t0 = time.perf_counter()
    for block in blocks:
        # blocks are already clipped to PROMPT_LEN (clip_blocks_to_prompt, run
        # locally before this call) and contiguous, so seq always already
        # covers exactly up to this block's start -- generate_block's own
        # prefix=seq[:block.start] slice is therefore a no-op here, but we
        # still pass the real `seq` rather than assume that, in case a future
        # caller doesn't pre-clip.
        cands = generate_block(seq, block, c, generate_fn, base_seed=seed)
        detailed = [scorer.score_candidate_detailed(cand, block) for cand in cands]
        scores = [s for s, _v in detailed]
        violations = [v for _s, v in detailed]
        pick = max(range(len(scores)), key=lambda i: scores[i])
        seq += cands[pick]
        rounds.append({
            "genes": list(block.genes), "start": block.start, "end": block.end,
            "picked": pick, "scores": [round(s, 6) for s in scores],
            "violations": violations,
            # No candidate was clean. Still a real ranking now (fewest
            # violations, PMI as tiebreak) -- but if this is true for EVERY
            # round, the arm is not selecting on k-mer evidence and the run
            # is a plain-sampling baseline. _run warns on that; it used to
            # pass silently as an index-0 tie. See BlockScorer.
            "all_violating": all(v > 0 for v in violations),
            # The specific failure that hid the 2026-11-13/14 run's breakage:
            # identical scores make max() return index 0 with no selection
            # having happened at all.
            "degenerate_tie": len(set(scores)) == 1 and len(scores) > 1,
        })
    torch.cuda.synchronize()
    block_kmer_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 2**30  # diagnostic: climbing across genomes in
                                                           # the same container would mean a real
                                                           # leak; a flat/context-length-correlated
                                                           # number (what's actually observed) means
                                                           # it's prefill's real peak, not a leak.
    es.close()
    del es
    torch.cuda.empty_cache()  # best-effort: release before a possible next genome reuses this container

    n_tokens = len(seq) - len(prompt)
    block_kmer_seq = seq[len(prompt):]
    return {
        "name": name,
        "accession": accession,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_requested": gpu,
        "price_per_hr": price,
        "model": model_name,
        "ckpt": ckpt,
        "load_s": round(load_s, 1),
        "n_tokens": n_tokens,
        "seed": seed,
        "c": c,
        "n_blocks": len(blocks),
        "block_kmer_s": round(block_kmer_s, 2),
        "block_kmer_toks_per_s": round(n_tokens / block_kmer_s, 1) if block_kmer_s > 0 else None,
        "n_rounds_all_violating": sum(1 for r in rounds if r["all_violating"]),
        "n_rounds_degenerate_tie": sum(1 for r in rounds if r["degenerate_tie"]),
        "peak_gpu_mem_gb": round(peak_gb, 2),
        "est_usd": round(block_kmer_s / 3600 * price, 5),
        "block_kmer_seq": block_kmer_seq,
        "rounds": rounds,
    }


def _mock_run(prompt: str, blocks: list, gene_to_cluster: dict, gene_coords: dict,
              tables: dict, c: int, seed: int) -> dict:
    """Same plumbing as run_block_kmer_prompt, no GPU: a fake sampler drawn
    from random in-frame-ish codons stands in for EvSpark, so the real
    round loop (generate_block, BlockScorer.score_candidate, argmax-pick)
    gets exercised, not just imported."""
    import numpy as np
    from aa_kmer_table import CODON_TABLE
    from block_pipeline import BlockScorer, generate_block

    scorer = BlockScorer(tables, gene_to_cluster, gene_coords)
    codons = [c for c in CODON_TABLE if CODON_TABLE[c] != "*"]

    def fake_generate_fn(prefix: str, n: int, s: int) -> str:
        rng = np.random.default_rng(s)
        n_codons = (n + 2) // 3
        return "".join(rng.choice(codons, size=n_codons))[:n]

    seq = prompt
    rounds = []
    for block in blocks:
        cands = generate_block(seq, block, c, fake_generate_fn, base_seed=seed)
        detailed = [scorer.score_candidate_detailed(cand, block) for cand in cands]
        scores = [s for s, _v in detailed]
        pick = max(range(len(scores)), key=lambda i: scores[i])
        seq += cands[pick]
        rounds.append({"genes": list(block.genes), "picked": pick,
                        "scores": [round(s, 6) for s in scores],
                        "violations": [v for _s, v in detailed],
                        "all_violating": all(v > 0 for _s, v in detailed),
                        "degenerate_tie": len(set(scores)) == 1 and len(scores) > 1})

    return {
        "n_tokens": len(seq) - len(prompt), "c": c, "n_blocks": len(blocks),
        "block_kmer_toks_per_s": None, "n_rounds": len(rounds),
        "block_kmer_seq": seq[len(prompt):], "rounds": rounds,
    }


def _results_path(tag: str) -> Path:
    return ROOT / f"block_kmer_bench_{GPU}{'_' + tag if tag else ''}.jsonl"


def _failures_path(tag: str) -> Path:
    return ROOT / f"block_kmer_failures_{GPU}{'_' + tag if tag else ''}.jsonl"


def _load_done_pairs(path: Path) -> set:
    """Resumability: one JSON object per line, so a process killed mid-write
    only ever corrupts the LAST (incomplete) line, not the whole file --
    every earlier line is independently valid. Returns {(accession, seed)}
    already present, so a rerun after a crash costs only what's missing."""
    done = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            done.add((r["accession"], r["seed"]))
        except (json.JSONDecodeError, KeyError):
            continue  # the one possibly-truncated last line from a prior crash -- skip, don't fail resume over it
    return done


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _run(model: str, ckpt: str, c: int, seeds: str, only: str, mock: bool, tag: str = "") -> None:
    """Shared body for both entrypoints below: `main` (modal run, real GPU
    calls allowed) and the plain `__main__` block (python ... --mock, no
    Modal connection at all -- same split kmer_fulllength_benchmark.py uses,
    so --mock stays a true $0/no-network test)."""
    import bio_scoring

    from block_pipeline import (build_tables, prepare_target_b, clip_blocks_to_prompt,
                                make_blocks, linearize_gene_coords)

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    genomes = {}
    for header, seq in read_fasta_records(str(GENOME_FASTA)):
        genomes[header.split()[0] if " " in header else header] = seq

    prompts = {}
    for header, seq in read_fasta_records(str(PROMPTS_FASTA)):
        prompts[header.split("__")[0]] = seq

    accessions = sorted(prompts)
    if only:
        prefixes = [p.strip() for p in only.split(",") if p.strip()]
        accessions = [a for a in accessions if any(a.startswith(p) for p in prefixes)]
    # Longest-first: if any genome is going to OOM, find out in the first
    # call instead of forty minutes into the batch, and the first failure
    # (if any) immediately tells you the real headroom on this GPU.
    accessions = sorted(accessions, key=lambda a: -len(genomes.get(a, "")))

    results_path, failures_path = _results_path(tag), _failures_path(tag)
    done_pairs = _load_done_pairs(results_path) if not mock else set()
    if done_pairs:
        print(f"resuming: {len(done_pairs)} (accession, seed) pair(s) already in {results_path.name}, will skip")

    print(f"block k-mer-arm run: {len(accessions)} target genomes, prompt_len={PROMPT_LEN}, "
          f"c={c}, seeds={seed_list}, mock={mock}")

    print("\nbuilding existing 20-genome reference tables (CPU-only, built once)...")
    tables = build_tables([REF_FASTA])

    results = []
    n_failed = 0
    for acc in accessions:
        remaining_seeds = [s for s in seed_list if (acc, s) not in done_pairs]
        if not remaining_seeds:
            print(f"skip {acc}: all seeds already in {results_path.name}")
            continue

        genome, prompt = genomes.get(acc), prompts.get(acc)
        if genome is None or prompt is None:
            print(f"skip {acc}: missing genome or prompt")
            continue
        if len(prompt) != PROMPT_LEN:
            print(f"skip {acc}: prompt length {len(prompt)} != expected {PROMPT_LEN}")
            continue

        gene_to_cluster, gene_coords, blocks, used_genome = prepare_target_b(
            acc, genome, str(CDS_FASTA), tables
        )
        if used_genome != genome:
            # prepare_target_b reverse-complements whole-minus-strand genomes --
            # should never happen for this run's 25-genome set (those 3 were
            # deliberately excluded), but fail loudly rather than silently
            # score against a frame the fixed prompt doesn't actually match.
            print(f"skip {acc}: prepare_target_b reverse-complemented this genome -- "
                  f"incompatible with the fixed external prompt, see prompts_description.txt section 6")
            continue

        # Where the prompt ACTUALLY sits. The old code clipped blocks at the
        # literal constant 300 instead, so it cut and scored blocks against
        # whichever gene sat near position 300 rather than the region the
        # model was really continuing -- see _prompt_offset.
        offset = _prompt_offset(prompt, genome)
        if offset is None:
            print(f"skip {acc}: prompt is not an exact substring of this genome -- "
                  f"the prompt and genome files disagree on the sequence itself, not just its framing")
            continue
        prompt_end = offset + len(prompt)
        lin_coords = linearize_gene_coords(gene_coords, len(genome), prompt_end)
        run_blocks = clip_blocks_to_prompt(make_blocks(lin_coords), lin_coords, 0)
        if not run_blocks:
            print(f"skip {acc}: no blocks remain ahead of prompt_end={prompt_end}")
            continue
        print(f"{acc}: prompt at genome offset {offset} (prompt_end={prompt_end}), "
              f"{len(run_blocks)} block(s) spanning {run_blocks[-1].end} bp of circular "
              f"continuation (genome_len={len(genome)})")

        for seed in remaining_seeds:
            name = f"{acc}_blockkmer_s{seed}"
            print(f"\n{name}: generating ({len(run_blocks)} blocks, genome_len={len(genome)})...")
            if mock:
                res = _mock_run(prompt, run_blocks, gene_to_cluster, lin_coords, tables, c, seed)
            else:
                # Catch ANY failure (OOM or otherwise) per genome rather than
                # letting it kill the whole run -- the real bug from
                # 2026-11-13 wasn't the OOM itself, it was that one crash
                # used to cost every OTHER genome's results too.
                try:
                    res = run_block_kmer_prompt.remote(
                        model, ckpt, name, acc, prompt, gene_to_cluster, lin_coords,
                        run_blocks, tables, c, seed, GPU, PRICE,
                    )
                except Exception as e:
                    n_failed += 1
                    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
                    _append_jsonl(failures_path, {
                        "accession": acc, "seed": seed, "name": name,
                        "error_type": type(e).__name__, "error": str(e)[:2000],
                        "n_blocks": len(run_blocks), "genome_len": len(genome),
                    })
                    continue
            res["accession"] = acc
            res["prompt_offset"] = offset
            res["prompt_end"] = prompt_end

            # --- CPU-local quality readout: generated continuation vs. the
            # REAL genome's own continuation at those exact positions (bio_scoring
            # is organism-agnostic, same tool speculative_benchmark.py/
            # kmer_fulllength_benchmark.py already use for this). ---
            gen_seq = res["block_kmer_seq"]
            real_seq = _circular_slice(genome, prompt_end, len(gen_seq))
            quality = {"generated": bio_scoring.score_sequence(gen_seq, organism_coding=True)}
            if len(real_seq) == len(gen_seq):
                quality["real"] = bio_scoring.score_sequence(real_seq, organism_coding=True)
                quality["generated_vs_real"] = bio_scoring.compare_sequences(
                    gen_seq, real_seq, organism_coding=True
                )
            res["quality"] = quality
            results.append(res)

            print(f"  {res.get('n_blocks')} blocks, {res.get('n_tokens')} tokens, "
                  f"toks/s={res.get('block_kmer_toks_per_s')}, peak_gpu_mem_gb={res.get('peak_gpu_mem_gb')}")

            # Selection health. The 2026-11-14 run produced complete,
            # plausible sequences while selecting candidate 0 in 98/98
            # rounds, because every candidate scored an identical -inf and
            # max() breaks ties by index. Nothing in the output said so.
            nb = res.get("n_blocks") or 0
            n_tie = res.get("n_rounds_degenerate_tie", 0)
            n_viol = res.get("n_rounds_all_violating", 0)
            picks = [r["picked"] for r in res.get("rounds", [])]
            print(f"  selection: {n_viol}/{nb} rounds with no clean candidate, "
                  f"{n_tie}/{nb} degenerate ties, picked={picks}")
            if nb and n_tie == nb:
                print(f"  WARNING {acc}: every round was a tie -- candidate 0 won "
                      f"by index, not by score. No k-mer selection from this genome.")
            gvr = quality.get("generated_vs_real")
            if gvr:
                print(f"  generated_vs_real: kmer_js={gvr['kmer_js_divergence']} "
                      f"codon_js={gvr['codon_js_divergence']}")

            # Append this ONE genome's result immediately -- a later genome
            # crashing (real OOM hit 2026-11-13, see BUILD_NOTES.md) used to
            # lose every earlier genome's completed work in the same batch
            # too, since results only got written once, after the whole loop
            # finished. This runs in the LOCAL entrypoint (this machine, not
            # the remote GPU container), so it isn't lost even if the
            # container that just generated it is torn down immediately after.
            if not mock:
                _append_jsonl(results_path, res)

    if mock:
        print(f"\nmock OK: {len(results)} target x seed combinations ran the real round loop, "
              f"no GPU, $0")
        return

    # Run-level guard. If selection never happened anywhere, this is a
    # plain-sampling baseline wearing the k-mer arm's name, and the --c
    # candidates were paid for and discarded. Say so loudly rather than
    # letting a downstream comparison be written up as "no improvement".
    tot_rounds = sum(r.get("n_blocks") or 0 for r in results)
    tot_ties = sum(r.get("n_rounds_degenerate_tie", 0) for r in results)
    tot_viol = sum(r.get("n_rounds_all_violating", 0) for r in results)
    if tot_rounds:
        print(f"selection summary: {tot_viol}/{tot_rounds} rounds had no clean "
              f"candidate, {tot_ties}/{tot_rounds} were degenerate ties")
        if tot_ties == tot_rounds:
            print("=" * 72)
            print("ERROR: NO SELECTION OCCURRED IN THIS ENTIRE RUN.")
            print("Every round scored all candidates identically, so candidate 0 won")
            print("by index in all of them. These sequences are an UNGUIDED baseline,")
            print("not a k-mer arm -- do not report a guided-vs-plain contrast from")
            print(f"them, and the --c {c} candidates per block were paid for and")
            print("thrown away.")
            print("=" * 72)

    print(f"\ndone: {len(results)} succeeded, {n_failed} failed this run -- "
          f"see {results_path.name}" + (f" and {failures_path.name}" if n_failed else ""))


@app.local_entrypoint()
def block_kmer(model: str = "evo2_7b", ckpt: str = "L27_g12_150M_s1",
                c: int = 3, seeds: str = "0", only: str = "", mock: bool = False, tag: str = ""):
    # Named block_kmer, not main -- `app` is the SAME modal.App shared with
    # speculative_benchmark.py (and transitively kmer_fulllength_benchmark.py),
    # which already registers its own local_entrypoint named "main" on it;
    # local_entrypoint names must be unique per App (hit this directly:
    # "Duplicate local entrypoint name: main" on first run).
    #
    # `tag` exists for running multiple batches CONCURRENTLY as separate
    # `modal run` processes (see BUILD_NOTES.md) -- without it, two parallel
    # batches both append to block_kmer_bench_L40S.jsonl, and since each
    # batch runs genomes sequentially from its OWN local entrypoint, two
    # DIFFERENT batches' processes could still interleave writes to one
    # shared file; separate per-tag files avoid that entirely.
    _run(model, ckpt, c, seeds, only, mock, tag)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--ckpt", default="L27_g12_150M_s1")
    ap.add_argument("--c", type=int, default=3)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not a.mock:
        raise SystemExit("run via `modal run block_kmer_benchmark.py` for a real GPU run; "
                          "use --mock here to test the plumbing with no GPU/no Modal connection")
    _run(a.model, a.ckpt, a.c, a.seeds, a.only, mock=True, tag=a.tag)
