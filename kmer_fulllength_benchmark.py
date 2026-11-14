"""Experiment 4: k-mer-guided ("SpecMER arm") full-length phiX174 generation.

The real comparison this script exists for: kmer-arm output vs. the native arm
already run in fulllength_bench_L40S.json (loaded here, see _load_baseline()).
Native is the thing we're trying to beat, not ground truth -- the real genome
comparison (kmer_vs_real) is context, not the target. Don't read a kmer(c=1)
run as a sanity check against native; chunked re-prefill sampling isn't
provably identical token-for-token to native's one-shot call even with
matching seeds, so "kmer vs. native" (both scored, both compared to real) is
the only comparison that means anything here -- not "kmer(c=1) == native".

Reuses fulllength_benchmark.py's exact prompt construction (genome[:prompt_len],
prompt_len=64, same reason: EvSpark's clean_prompt() hard-floors prompts at 64bp)
and the same seeds [0, 1, 2], model (evo2_7b), and checkpoint (L27_g12_150M_s1) --
so this script's "kmer" column is directly comparable row-for-row against
fulllength_bench_<GPU>.json's "native"/"speculative" columns, not a separate
experiment with its own confounds.

Mechanics (the "k-mer arm", ported from Capstone/kmer-evo2/generate_evspark.py's
design, but using the plain EvSpark.generate_native() call already proven in this
repo's speculative_benchmark.py -- not kmer-evo2's untested CandidateGenerator
state-reuse path):
  Each round, grow the sequence by --chunk-len letters. Instead of one Evo2 call,
  make --c calls with distinct seeds (same candidate_seeds() derivation as
  kmer-evo2), score every candidate with the k-mer table (kmer_table.KmerTable.
  score, mode=--score-mode), keep the highest-scoring one, repeat until
  --n-tokens letters have been generated.

Cost note (why this is NOT run here): --c candidates per round means --c forward
passes instead of 1, and because generate_native() re-encodes the whole prompt+
generated-so-far context on every call (no persisted KV state across chunks --
that reuse is exactly what kmer-evo2's untested CandidateGenerator path would add
instead), total compute grows faster than fulllength_benchmark.py's single-shot
call: roughly O(c * genome_len^2 / chunk_len) token-equivalents vs O(genome_len).
At the defaults below (c=3, chunk_len=512, genome_len~5.3kb) that is a rough 5-15x
multiple on top of fulllength_benchmark.py's per-seed GPU time/cost -- tune --c /
--chunk-len down before spending real money on Modal if that multiple matters.

k-mer table: dependencies/microviridae_k346.npz, built by build_kmer_table.py
from 20 OTHER Microviridae genomes (~105.7kb; close coliphage relatives plus
deliberately divergent relatives across other hosts/subfamilies, for diversity
of signal, not just to dilute sparsity) -- never phiX174 itself. See
build_kmer_table.py's docstring for the full list, the circularity rationale,
and why k=8 was dropped (ks=[3, 4, 6] only) even after the expansion.

EvSpark API status: generate_native()'s return shape (.text/.wall_s/.tok_s) is
NOT a guess -- fulllength_bench_L40S.json already contains real GPU output from
this exact call (via speculative_benchmark.run_prompt), so that surface is
proven. What's new and unverified here is calling generate_native() repeatedly
in a loop on a context that keeps growing past the original prompt (every prior
real run made exactly one generate_native() call per sequence, at the original
prompt length) -- untested whether that's correct/efficient/stable at ~5kb of
context by round 11.

Run:  BENCH_GPU=L40S GPU_PRICE_PER_HR=1.95 modal run kmer_fulllength_benchmark.py
Options: --prompt-len 64 --n-tokens 0 --c 3 --chunk-len 512 --score-mode paper
         --table dependencies/microviridae_k346.npz

score_mode="aa_gated" (2026-11-04): amino-acid-guided arm, see aa_kmer_table.py
and BUILD_NOTES.md. For each round, translates every candidate in the frame(s)
of whichever real phiX174 gene(s) overlap that round's span (lookup against
PHIX174_GENES_NOWRAP below -- exact, not a gene caller, since generation is a
literal linear continuation of the real genome's own coordinates). A candidate
that introduces a premature in-frame stop codon in ANY active gene is hard-
gated out (composite score -1e6 - violations, so it can never outrank a clean
candidate, but ties among all-broken rounds still break toward fewer
violations rather than arbitrarily). Gate-survivors are ranked by mean amino-
acid k-mer score (mode="relative", ks=[2,3], equal weight per k -- avoids the
DNA arm's k=3-domination flaw) across all active frames. Needs --aa-table
(default dependencies/microviridae_aa_k23.npz, built by build_aa_kmer_table.py
from real CDS translations of the same 20 reference genomes, never phiX174
itself). Rounds whose span has no overlapping gene (small intergenic gaps
only, max 110bp vs. chunk_len=512 so effectively unreachable at default
chunk_len) fall back to the DNA relative-mode scorer for that round.
Writes: kmer_fulllength_bench_<GPU>.json (DNA modes) or
        kmer_fulllength_bench_<GPU>_aa.json (score_mode="aa_gated")

Mock-test the plumbing (no GPU, $0):
  python kmer_fulllength_benchmark.py --mock
"""
import json
import os
import time
from pathlib import Path

import modal

from speculative_benchmark import GPU, PRICE, app, read_fasta_records, image, EVSPARK_DIR

# speculative_benchmark.py is imported at module level by run_kmer_prompt's own
# module (this file) -- the shared `image` above never bakes that sibling file
# in, so without this the remote container crash-loops on ModuleNotFoundError.
image = image.add_local_python_source("speculative_benchmark")

PHIX174_FASTA = Path(__file__).parent / "phix174.fasta"
if not PHIX174_FASTA.exists():
    PHIX174_FASTA = Path(__file__).parent / "dependencies" / "phix174.fasta"
DEFAULT_TABLE = Path(__file__).parent / "dependencies" / "microviridae_k346.npz"
DEFAULT_AA_TABLE = Path(__file__).parent / "dependencies" / "microviridae_aa_k23.npz"
BASELINE_PATH = Path(__file__).parent / "fulllength_bench_L40S.json"
SEEDS = [0, 1, 2]

# phiX174 genes, 0-based half-open, single-segment (non-origin-wrapping) only --
# A/A*/B wrap the origin and are skipped here the same way genome_scoring.
# orf_closure_check skips them for prompt_len>0 (the wrap segment doesn't exist
# in a linear one-shot continuation). Source: genome_scoring.PHIX174_GENES.
PHIX174_GENES_NOWRAP = [
    (50, 221),    # K
    (132, 393),   # C
    (389, 848),   # D
    (567, 843),   # E
    (847, 964),   # J
    (1000, 2284), # F
    (2394, 2922), # G
    (2930, 3917), # H
]


def _load_baseline() -> dict:
    """name -> result dict from the already-collected native/speculative run
    (fulllength_bench_L40S.json), so the kmer arm can be compared against the
    actual native sequence/metrics it's trying to beat, not just the real genome."""
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text())
    return {r["name"]: r for r in data["results"]}


def candidate_seeds(base_seed: int, round_index: int, c: int):
    """Distinct, stable seeds. Candidate 0 of the k-mer arm uses the SAME seed a
    plain --c=1 baseline would use for this round, so c=1 reproduces the baseline
    exactly (matches kmer-evo2/generate_evspark.py's convention)."""
    return [int(base_seed * 1_000_003 + round_index * 1009 + j) for j in range(c)]


@app.function(image=image, gpu=GPU, timeout=3 * 60 * 60)
def run_kmer_prompt(model_name: str, ckpt: str, name: str, region: str, prompt: str,
                     n_tokens: int, seed: int, gpu: str, price: float,
                     chunk_len: int, c: int, score_mode: str, table_bytes: bytes,
                     aa_table_bytes: bytes | None = None) -> dict:
    import io
    import numpy as np
    import torch
    from evspark import EvSpark
    from evspark.api import clean_prompt

    # --- minimal, self-contained k-mer table + SpecMER score (mirrors kmer_table.py;
    # duplicated here rather than imported so this Modal container doesn't need
    # kmer_table.py baked into its image). ---
    z = np.load(io.BytesIO(table_bytes))
    ks = [int(k) for k in z["ks"]]
    probs = {}
    for k in ks:
        counts = z[f"c{k}"]
        total = counts.sum()
        probs[k] = (counts / total) if total > 0 else counts
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    bases = "ACGT"
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(bases):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i

    def encode(s):
        return lut[np.frombuffer(s.encode(), dtype=np.uint8)]

    def kmer_indices(arr, k):
        n = len(arr) - k + 1
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        idx = np.zeros(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        for j in range(k):
            seg = arr[j: j + n]
            valid &= seg >= 0
            idx = idx * 4 + np.where(seg >= 0, seg, 0)
        return idx[valid]

    def kmer_score(seq, context, mode=None):
        L = len(seq)
        if L == 0:
            return 0.0
        mode = mode or score_mode  # mode=None -> use the outer score_mode (DNA modes only;
        # the aa_gated fallback below passes mode="relative" explicitly, since
        # score_mode itself is "aa_gated" there, not a valid DNA mode)
        total = 0.0
        for k in ks:
            ctx = context[-(k - 1):] if (k > 1 and context) else ""
            idx = kmer_indices(encode(ctx + seq), k)
            if len(idx) == 0:
                continue
            p = probs[k][idx]
            if mode == "paper":
                total += p.sum()
            elif mode == "relative":
                total += (p * 4 ** k).sum()
            elif mode == "loglik":
                total += np.log(p + 1e-12).sum()
            else:
                raise ValueError(mode)
        return total / L

    # --- amino-acid gated scorer (mirrors aa_kmer_table.py; duplicated here for
    # the same reason as the DNA scorer above: this Modal container doesn't have
    # aa_kmer_table.py baked into its image). ---
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

    def translate(dna):
        dna = dna.upper()
        n = len(dna) - (len(dna) % 3)
        return "".join(CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, n, 3))

    if score_mode == "aa_gated":
        AA = "ACDEFGHIKLMNPQRSTVWY"
        aa_lut = np.full(256, -1, dtype=np.int64)
        for i, ch in enumerate(AA):
            aa_lut[ord(ch)] = i

        def aa_encode(s):
            return aa_lut[np.frombuffer(s.encode(), dtype=np.uint8)]

        def aa_kmer_indices(arr, k):
            n = len(arr) - k + 1
            if n <= 0:
                return np.empty(0, dtype=np.int64)
            idx = np.zeros(n, dtype=np.int64)
            valid = np.ones(n, dtype=bool)
            for j in range(k):
                seg = arr[j: j + n]
                valid &= seg >= 0
                idx = idx * 20 + np.where(seg >= 0, seg, 0)
            return idx[valid]

        az = np.load(io.BytesIO(aa_table_bytes))
        aa_ks = [int(k) for k in az["ks"]]
        aa_probs = {}
        for k in aa_ks:
            counts = az[f"c{k}"]
            total = counts.sum()
            aa_probs[k] = (counts / total) if total > 0 else counts

        def aa_score(aa_seq):
            L = len(aa_seq)
            if L == 0:
                return 0.0
            total = 0.0
            for k in aa_ks:
                idx = aa_kmer_indices(aa_encode(aa_seq), k)
                if len(idx) == 0:
                    continue
                total += (aa_probs[k][idx] * 20 ** k).sum()
            return total / L

        def aa_gated_score(dna_cand, lo, hi):
            """Hard-gates on a premature stop codon in any gene frame active over
            [lo, hi) (checked against the genome's own real gene coordinates --
            this only works because the generation is a literal linear
            continuation of phiX174's own coordinate frame, same assumption
            genome_scoring.orf_closure_check relies on). Among gate-survivors,
            ranks by mean amino-acid k-mer "relative" score across active frames.
            Returns (composite_score, active: bool) -- active=False means no gene
            overlaps this span (caller should fall back to the DNA scorer).

            For each active gene, translation starts at the first codon boundary
            of THAT GENE's own frame at or after max(lo, gene_start) -- never
            before the gene's real start, even if that frame's phase happens to
            align earlier in this chunk (that earlier region belongs to a
            different gene/frame, not this one, and checking it there produced
            false positives on the real genome during validation). This also
            means no cross-chunk context is needed: codon_abs_start is always
            >= lo, so translation only ever reads forward within dna_cand -- at
            most 2bp of genuine straddling-codon coverage is lost at a chunk's
            leading edge, the same order of boundary loss kmer_table.py already
            accepts for DNA k-mer context."""
            active = [(s, e) for (s, e) in PHIX174_GENES_NOWRAP if e > lo and s < hi]
            if not active:
                return None, False
            violations = 0
            continuous = []
            used_any = False
            for s, e in active:
                base = max(lo, s)
                rem = (base - s) % 3
                codon_abs_start = base + ((3 - rem) % 3)
                if codon_abs_start >= hi:
                    continue
                aa = translate(dna_cand[codon_abs_start - lo : hi - lo])
                expected_idx = (e - codon_abs_start) // 3 - 1 if e <= hi else None
                if expected_idx is not None and expected_idx < 0:
                    continue
                check_region = aa[: expected_idx + 1] if expected_idx is not None else aa
                local_viol = sum(1 for i, ch in enumerate(check_region) if ch == "*" and i != expected_idx)
                violations += local_viol
                gene_region = check_region[:expected_idx] if expected_idx is not None else check_region
                if gene_region:
                    continuous.append(aa_score(gene_region.replace("*", "")))
                used_any = True
            if not used_any:
                return None, False
            if violations > 0:
                return -1e6 - violations, True
            return (sum(continuous) / len(continuous) if continuous else 0.0), True

    t0 = time.perf_counter()
    es = EvSpark.load(ckpt=ckpt, model_name=model_name, device="cuda:0")
    load_s = time.perf_counter() - t0

    prompt = clean_prompt(prompt)

    # --- warm-up, untimed (see speculative_benchmark.run_prompt for rationale) ---
    WARMUP_TOKENS = 32
    es.generate_native(prompt, n_tokens=WARMUP_TOKENS, greedy=False, seed=seed,
                        temperature=1.0, top_k=4)
    torch.cuda.synchronize()

    seq = prompt
    rounds = []
    r = 0
    t0 = time.perf_counter()
    while len(seq) - len(prompt) < n_tokens:
        n = min(chunk_len, n_tokens - (len(seq) - len(prompt)))
        lo = len(seq)
        hi = lo + n
        seeds_j = candidate_seeds(seed, r, c)
        cands = [es.generate_native(seq, n_tokens=n, greedy=False, seed=s,
                                     temperature=1.0, top_k=4).text for s in seeds_j]
        if score_mode == "aa_gated":
            results = [aa_gated_score(cand, lo, hi) for cand in cands]
            if results[0][1]:  # active gene(s) over this span -- use the gated score
                scores = [res[0] for res in results]
            else:  # no gene overlaps this span (rare, small intergenic gaps only) -- fall back to DNA relative score
                scores = [kmer_score(cand, seq, mode="relative") for cand in cands]
        else:
            scores = [kmer_score(cand, seq) for cand in cands]
        pick = int(np.argmax(scores))
        seq += cands[pick]
        rounds.append({"round": r, "picked": pick, "scores": [round(float(x), 6) for x in scores]})
        r += 1
    torch.cuda.synchronize()
    kmer_s = time.perf_counter() - t0
    es.close()

    kmer_seq = seq[len(prompt):]
    return {
        "name": name,
        "region": region,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_requested": gpu,
        "price_per_hr": price,
        "model": model_name,
        "ckpt": ckpt,
        "load_s": round(load_s, 1),
        "n_tokens": n_tokens,
        "seed": seed,
        "c": c,
        "chunk_len": chunk_len,
        "score_mode": score_mode,
        "kmer_s": round(kmer_s, 2),
        "kmer_toks_per_s": round(n_tokens / kmer_s, 1) if kmer_s > 0 else None,
        "n_rounds": len(rounds),
        "est_usd_kmer": round(kmer_s / 3600 * price, 5),
        "kmer_seq": kmer_seq,
        "rounds": rounds,
    }


@app.local_entrypoint()
def kmer_fulllength(model: str = "evo2_7b", ckpt: str = "L27_g12_150M_s1",
                     prompt_len: int = 64, n_tokens: int = 0,
                     chunk_len: int = 512, c: int = 3, score_mode: str = "paper",
                     table: str = str(DEFAULT_TABLE), aa_table: str = str(DEFAULT_AA_TABLE),
                     seeds: str = "0,1,2", mock: bool = False):
    import bio_scoring
    import genome_scoring

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    ref_name, genome = read_fasta_records(str(PHIX174_FASTA))[0]
    genome_len = len(genome)
    tokens = n_tokens if n_tokens > 0 else genome_len - prompt_len
    prompt = genome[:prompt_len]
    table_bytes = Path(table).read_bytes()
    aa_table_bytes = Path(aa_table).read_bytes() if score_mode == "aa_gated" else None
    baseline = _load_baseline()
    print(f"phiX174 k-mer-arm run: {ref_name}, genome_len={genome_len}, "
          f"prompt_len={prompt_len}, n_tokens={tokens}, c={c}, chunk_len={chunk_len}, "
          f"score_mode={score_mode}, table={table}, "
          f"baseline={'loaded (' + str(len(baseline)) + ' entries)' if baseline else 'NOT FOUND'}")

    if mock:
        _mock_test(prompt, tokens, chunk_len, c, table_bytes, score_mode)
        return

    results = []
    for seed in seed_list:
        name = f"phix174_fulllength_kmer_s{seed}"
        print(f"\n{name}: generating...")
        res = run_kmer_prompt.remote(model, ckpt, name, "viral_coding_fulllength", prompt,
                                      tokens, seed, GPU, PRICE, chunk_len, c, score_mode, table_bytes,
                                      aa_table_bytes)

        kmer_seq = res["kmer_seq"]
        real_seq = genome[prompt_len: prompt_len + tokens]
        has_real = len(real_seq) == tokens

        quality = {
            "kmer": bio_scoring.score_sequence(kmer_seq, organism_coding=True),
            "genome_kmer": genome_scoring.score_genome(prompt + kmer_seq, prompt_len=prompt_len),
        }
        if has_real:
            quality["real"] = bio_scoring.score_sequence(real_seq, organism_coding=True)
            quality["kmer_vs_real"] = bio_scoring.compare_sequences(kmer_seq, real_seq, organism_coding=True)
            quality["genome_real"] = genome_scoring.score_genome(genome, prompt_len=0)

        # --- the real comparison: kmer arm vs. the native arm it's trying to
        # beat (fulllength_bench_L40S.json), not vs. speculative and not an
        # identity check against native. ---
        base = baseline.get(f"phix174_fulllength_s{seed}")
        if base:
            native_seq = base["native_seq"]
            quality["native_baseline"] = base["quality"]["native"]          # precomputed, reused as-is
            quality["kmer_vs_native"] = bio_scoring.compare_sequences(kmer_seq, native_seq, organism_coding=True)
            quality["genome_native_baseline"] = base["quality"]["genome_native"]
        else:
            print(f"  warning: no baseline entry for seed {seed} in {BASELINE_PATH.name} "
                  f"-- kmer_vs_native comparison skipped")

        res["quality"] = quality
        results.append(res)

        gk = quality["genome_kmer"]
        print(f"  kmer {res['kmer_toks_per_s']} tok/s  rounds={res['n_rounds']}")
        print(f"  orf_closure: kmer {gk['orf_closure']['n_closed']}/{gk['orf_closure']['n_scored']}"
              + (f"  native {quality['genome_native_baseline']['orf_closure']['n_closed']}/"
                 f"{quality['genome_native_baseline']['orf_closure']['n_scored']}" if base else ""))
        print(f"  gc_skew sign changes: kmer {gk['gc_skew']['n_sign_changes']}  "
              + (f"native {quality['genome_native_baseline']['gc_skew']['n_sign_changes']}  " if base else "")
              + f"(real {quality.get('genome_real', {}).get('gc_skew', {}).get('n_sign_changes')})")
        print(f"  terminator hits: kmer {gk['terminators']['n_hits']}"
              + (f"  native {quality['genome_native_baseline']['terminators']['n_hits']}" if base else ""))
        real_q = quality.get("real", {})
        print(f"  pyrodigal: kmer n_genes={quality['kmer'].get('n_genes')} "
              f"coding_density={quality['kmer'].get('coding_density')}  "
              + (f"native n_genes={quality['native_baseline'].get('n_genes')} "
                 f"coding_density={quality['native_baseline'].get('coding_density')}  " if base else "")
              + f"real n_genes={real_q.get('n_genes')} coding_density={real_q.get('coding_density')}")
        if base:
            print(f"  kmer_vs_native: kmer_js_divergence={quality['kmer_vs_native']['kmer_js_divergence']} "
                  f"codon_js_divergence={quality['kmer_vs_native']['codon_js_divergence']}")

    suffix = "_aa" if score_mode == "aa_gated" else ""
    out_path = f"kmer_fulllength_bench_{GPU}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model, "ckpt": ckpt, "prompt_len": prompt_len, "n_tokens": tokens,
            "seeds": seed_list, "genome_len": genome_len, "gpu": GPU, "price_per_hr": PRICE,
            "c": c, "chunk_len": chunk_len, "score_mode": score_mode, "table": str(table),
            "aa_table": str(aa_table) if score_mode == "aa_gated" else None,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


def _mock_test(prompt, tokens, chunk_len, c, table_bytes, score_mode):
    """Exercise the k-mer scoring + round loop without a GPU, using a tiny fake
    sampler drawn from the table itself (order-1 Markov) so the plumbing (seed
    derivation, chunking, best-of-c picking, sequence assembly) gets a real test.
    NOT a stand-in for the actual EvSpark call -- just checks nothing crashes and
    the output types/shapes are sane."""
    import io

    import numpy as np

    z = np.load(io.BytesIO(table_bytes))
    ks = [int(k) for k in z["ks"]]
    probs = {}
    for k in ks:
        counts = z[f"c{k}"]
        total = counts.sum()
        probs[k] = (counts / total) if total > 0 else counts
    bases = "ACGT"
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(bases):
        lut[ord(ch)] = i
        lut[ord(ch.lower())] = i

    def encode(s):
        return lut[np.frombuffer(s.encode(), dtype=np.uint8)]

    def kmer_indices(arr, k):
        n = len(arr) - k + 1
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        idx = np.zeros(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        for j in range(k):
            seg = arr[j: j + n]
            valid &= seg >= 0
            idx = idx * 4 + np.where(seg >= 0, seg, 0)
        return idx[valid]

    def kmer_score(seq, context):
        L = len(seq)
        if L == 0:
            return 0.0
        total = 0.0
        for k in ks:
            ctx = context[-(k - 1):] if (k > 1 and context) else ""
            idx = kmer_indices(encode(ctx + seq), k)
            if len(idx) == 0:
                continue
            p = probs[k][idx]
            if score_mode == "paper":
                total += p.sum()
            elif score_mode == "relative":
                total += (p * 4 ** k).sum()
            elif score_mode == "loglik":
                total += np.log(p + 1e-12).sum()
        return total / L

    rng = np.random.default_rng(0)

    def fake_sampler(context, n, seed):
        local_rng = np.random.default_rng(seed)
        return "".join(bases[i] for i in local_rng.integers(0, 4, size=n))

    seq = prompt
    rounds = []
    r = 0
    while len(seq) - len(prompt) < tokens:
        n = min(chunk_len, tokens - (len(seq) - len(prompt)))
        seeds_j = candidate_seeds(0, r, c)
        cands = [fake_sampler(seq, n, s) for s in seeds_j]
        scores = [kmer_score(cand, seq) for cand in cands]
        pick = int(np.argmax(scores))
        seq += cands[pick]
        rounds.append({"round": r, "picked": pick, "scores": scores})
        r += 1

    assert len(seq) == len(prompt) + tokens, "mock run produced wrong length"
    assert all(ch in bases for ch in seq[len(prompt):]), "mock run produced non-ACGT letters"
    print(f"mock OK: {len(rounds)} rounds, final length {len(seq)} "
          f"(prompt {len(prompt)} + {tokens} generated), c={c}, chunk_len={chunk_len}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--n-tokens", type=int, default=0)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--c", type=int, default=3)
    ap.add_argument("--score-mode", default="paper", choices=["paper", "relative", "loglik"])
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    a = ap.parse_args()
    if not a.mock:
        raise SystemExit("run via `modal run kmer_fulllength_benchmark.py` for a real GPU run; "
                          "use --mock here to test the plumbing")

    ref_name, genome = read_fasta_records(str(PHIX174_FASTA))[0]
    tokens = a.n_tokens if a.n_tokens > 0 else len(genome) - a.prompt_len
    prompt = genome[:a.prompt_len]
    table_bytes = Path(a.table).read_bytes()
    _mock_test(prompt, tokens, a.chunk_len, a.c, table_bytes, a.score_mode)
