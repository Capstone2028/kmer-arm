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
Writes: kmer_fulllength_bench_<GPU>.json

Mock-test the plumbing (no GPU, $0):
  python kmer_fulllength_benchmark.py --mock
"""
import json
import os
import time
from pathlib import Path

import modal

from speculative_benchmark import GPU, PRICE, app, read_fasta_records, image, EVSPARK_DIR

PHIX174_FASTA = Path(__file__).parent / "phix174.fasta"
if not PHIX174_FASTA.exists():
    PHIX174_FASTA = Path(__file__).parent / "dependencies" / "phix174.fasta"
DEFAULT_TABLE = Path(__file__).parent / "dependencies" / "microviridae_k346.npz"
BASELINE_PATH = Path(__file__).parent / "fulllength_bench_L40S.json"
SEEDS = [0, 1, 2]


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
                     chunk_len: int, c: int, score_mode: str, table_bytes: bytes) -> dict:
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
            else:
                raise ValueError(score_mode)
        return total / L

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
        seeds_j = candidate_seeds(seed, r, c)
        cands = [es.generate_native(seq, n_tokens=n, greedy=False, seed=s,
                                     temperature=1.0, top_k=4).text for s in seeds_j]
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
                     table: str = str(DEFAULT_TABLE), seeds: str = "0,1,2",
                     mock: bool = False):
    import bio_scoring
    import genome_scoring

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    ref_name, genome = read_fasta_records(str(PHIX174_FASTA))[0]
    genome_len = len(genome)
    tokens = n_tokens if n_tokens > 0 else genome_len - prompt_len
    prompt = genome[:prompt_len]
    table_bytes = Path(table).read_bytes()
    baseline = _load_baseline()
    print(f"phiX174 k-mer-arm run: {ref_name}, genome_len={genome_len}, "
          f"prompt_len={prompt_len}, n_tokens={tokens}, c={c}, chunk_len={chunk_len}, "
          f"table={table}, baseline={'loaded (' + str(len(baseline)) + ' entries)' if baseline else 'NOT FOUND'}")

    if mock:
        _mock_test(prompt, tokens, chunk_len, c, table_bytes, score_mode)
        return

    results = []
    for seed in seed_list:
        name = f"phix174_fulllength_kmer_s{seed}"
        print(f"\n{name}: generating...")
        res = run_kmer_prompt.remote(model, ckpt, name, "viral_coding_fulllength", prompt,
                                      tokens, seed, GPU, PRICE, chunk_len, c, score_mode, table_bytes)

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

    out_path = f"kmer_fulllength_bench_{GPU}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model, "ckpt": ckpt, "prompt_len": prompt_len, "n_tokens": tokens,
            "seeds": seed_list, "genome_len": genome_len, "gpu": GPU, "price_per_hr": PRICE,
            "c": c, "chunk_len": chunk_len, "score_mode": score_mode, "table": str(table),
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
