"""Experiment 5: per-codon amino-acid k-mer GUIDED decoding, full-length phiX174.

Different mechanism from kmer_fulllength_benchmark.py's score_mode="aa_gated"
(best-of-c: generate c whole 512bp chunks, keep whichever scores best). This
script reweights the model's OWN per-token sampling distribution live, one
nucleotide at a time, using EvSpark's exported low-level primitives instead of
its generate_native()/generate() wrappers:
  - evspark.specdec.block.driver.prefill / step_forward_reference -- raw
    per-token logits, one call per generated base, same KV-cache state
    (inference_params) persisted for the ENTIRE generation (no chunk
    boundaries, no re-prefill -- incidentally also fixes the "resend full
    context every round" cost problem kmer_fulllength_benchmark.py has).
  - evspark.specdec.transforms.apply_transform -- logits -> sampling probs
    (same T/top_k transform generate_native() uses internally).
  - evspark.specdec.verifier.sample_categorical -- the actual sample.

Why guidance happens per-CODON, not per-base: a k-mer table has nothing to
score until a complete k-mer word exists. A single nucleotide isn't a complete
amino-acid word, so bases 1-2 of every codon are sampled completely unguided --
nothing to gate or weight yet. Base 3 (the "closing" base) is the one moment a
new, complete amino acid becomes known; THAT'S where the amino-acid k-mer
table has something to say, and -- since bases 1-2 are already fixed by then --
there's no lookahead needed: at most 4 candidate completions, each mapping to
one fully-determined amino acid or a stop.

2026-11-04 fix: codon-closing steps now gate/reweight on the model's TRUE,
untruncated softmax over the 4 bases (apply_transform(..., top_k=None)), not
the top_k=4-truncated distribution every other step uses. First version
truncated to top-4 BEFORE gating, so a correct-but-not-natively-favored base
(above all, the real stop codon at a gene's end -- the model essentially
never ranks that in its own top-4 unprompted, same reason native/speculative/
best-of-c all show 0/7 orf_closure too) was already zero by the time the gate
ran, making the "force" a no-op. First GPU run confirmed this: forced_stop=0/8,
fallback=8/8 -- every single forcing opportunity degenerated to a no-op.
Non-closing steps are unaffected, still top_k=4 (native sampling convention,
comparability with the other arms).

Gating/weighting logic lives in aa_guided_decode.py (pure, GPU-free, unit-
tested against the real genome -- see BUILD_NOTES.md) and is duplicated inline
below for the Modal container (same reason kmer_fulllength_benchmark.py
duplicates kmer_table.py: no local repo files are baked into the image).
Design decisions (forced terminal stop at each gene's real end; overlapping
genes closing at the same position require BOTH to pass, weights multiplied
not averaged) are documented in aa_guided_decode.py's docstring.

Same prompt construction, seeds, model, checkpoint as kmer_fulllength_benchmark.py
and fulllength_benchmark.py -- directly comparable row-for-row.

Needs dependencies/microviridae_aa_k23.npz (build_aa_kmer_table.py). No DNA
k-mer table used here at all -- bases 1-2 of every codon are unguided by
design, not a fallback.

Run:  BENCH_GPU=L40S GPU_PRICE_PER_HR=1.95 modal run kmer_guided_fulllength_benchmark.py
Options: --prompt-len 64 --n-tokens 0 --seeds 0,1,2
         --aa-table dependencies/microviridae_aa_k23.npz
Writes: kmer_fulllength_bench_<GPU>_guided.json
"""
import json
import os
import time
from pathlib import Path

import modal

from speculative_benchmark import GPU, PRICE, app, read_fasta_records, image, EVSPARK_DIR

image = image.add_local_python_source("speculative_benchmark")

PHIX174_FASTA = Path(__file__).parent / "phix174.fasta"
if not PHIX174_FASTA.exists():
    PHIX174_FASTA = Path(__file__).parent / "dependencies" / "phix174.fasta"
DEFAULT_AA_TABLE = Path(__file__).parent / "dependencies" / "microviridae_aa_k23.npz"
BASELINE_PATH = Path(__file__).parent / "fulllength_bench_L40S.json"
SEEDS = [0, 1, 2]

PHIX174_GENES_NOWRAP = [
    (50, 221),    # C
    (132, 393),   # D
    (389, 848),   # F
    (567, 843),   # G
    (847, 964),   # H
    (1000, 2284), # E
    (2394, 2922), # J
    (2930, 3917), # K
]


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text())
    return {r["name"]: r for r in data["results"]}


@app.function(image=image, gpu=GPU, timeout=3 * 60 * 60)
def run_guided_prompt(model_name: str, ckpt: str, name: str, region: str, prompt: str,
                       n_tokens: int, seed: int, gpu: str, price: float,
                       aa_table_bytes: bytes) -> dict:
    import io

    import numpy as np
    import torch
    from evspark import EvSpark
    from evspark.api import clean_prompt
    from evspark.specdec.block.driver import prefill, step_forward_reference
    from evspark.specdec.transforms import apply_transform
    from evspark.specdec.verifier import sample_categorical

    AA = "ACDEFGHIKLMNPQRSTVWY"
    _AA_IDX = {ch: i for i, ch in enumerate(AA)}
    BASES = "ACGT"
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

    def translate_codons(dna):
        dna = dna.upper()
        n = len(dna) - (len(dna) % 3)
        return "".join(CODON_TABLE.get(dna[i:i + 3], "X") for i in range(0, n, 3))

    az = np.load(io.BytesIO(aa_table_bytes))
    aa_ks = [int(k) for k in az["ks"]]
    aa_probs = {}
    for k in aa_ks:
        counts = az[f"c{k}"]
        total = counts.sum()
        aa_probs[k] = (counts / total) if total > 0 else counts

    def aa_transition_weight(context, candidate):
        total = 0.0
        n = 0
        for k in aa_ks:
            if len(context) < k - 1:
                continue
            window = (context[-(k - 1):] if k > 1 else "") + candidate
            idx = 0
            for ch in window:
                idx = idx * 20 + _AA_IDX[ch]
            total += aa_probs[k][idx] * 20 ** k
            n += 1
        return total / n if n > 0 else 1e-6

    def seed_gene_aa(prompt_seq, prompt_len):
        gene_aa = {}
        for gi, (s, e) in enumerate(PHIX174_GENES_NOWRAP):
            if s < prompt_len:
                codon_end = s + 3 * ((min(e, prompt_len) - s) // 3)
                gene_aa[gi] = translate_codons(prompt_seq[s:codon_end])
            else:
                gene_aa[gi] = ""
        return gene_aa

    def closing_genes(pos):
        return [(gi, s, e) for gi, (s, e) in enumerate(PHIX174_GENES_NOWRAP)
                if s <= pos < e and (pos - s) % 3 == 2]

    def candidate_weights(seq, pos, closing, gene_aa):
        result = {b: (True, 1.0) for b in BASES}
        for gi, s, e in closing:
            is_final = pos == e - 1
            for b in BASES:
                allowed, weight = result[b]
                if not allowed:
                    continue
                codon = seq[pos - 2] + seq[pos - 1] + b
                aa = CODON_TABLE.get(codon, "X")
                if is_final:
                    if aa != "*":
                        result[b] = (False, weight)
                else:
                    if aa == "*" or aa == "X":
                        result[b] = (False, weight)
                    else:
                        w = aa_transition_weight(gene_aa[gi], aa)
                        result[b] = (True, weight * w)
        return result

    def advance_gene_aa(seq_with_new_base, pos, closing, gene_aa):
        for gi, s, e in closing:
            codon = seq_with_new_base[pos - 2: pos + 1]
            aa = CODON_TABLE.get(codon, "X")
            if aa not in ("*", "X"):
                gene_aa[gi] = gene_aa[gi] + aa

    byte_id = {b: ord(b) for b in BASES}

    t0 = time.perf_counter()
    es = EvSpark.load(ckpt=ckpt, model_name=model_name, device="cuda:0")
    load_s = time.perf_counter() - t0

    prompt = clean_prompt(prompt)
    prompt_len = len(prompt)

    WARMUP_TOKENS = 32
    es.generate_native(prompt, n_tokens=WARMUP_TOKENS, greedy=False, seed=seed,
                        temperature=1.0, top_k=4)
    torch.cuda.synchronize()

    rng = np.random.default_rng(seed)
    model = es.model
    _, prompt_ids = es._tokenize(prompt)
    device = prompt_ids.device

    gene_aa = seed_gene_aa(prompt, prompt_len)
    n_closing_steps = 0
    n_forced_stop = 0
    n_gate_rejections = 0  # candidate bases zeroed out across all closing steps
    n_fallback = 0  # steps where every candidate got zeroed and we fell back unbiased

    t0 = time.perf_counter()
    ip = model.initialize_inference_params(max_seqlen=prompt_len + n_tokens + 2)
    with torch.inference_mode():
        logits = prefill(model, prompt_ids, ip)
    last = logits[0, -1]

    seq_chars = list(prompt)
    emitted = np.empty(n_tokens, dtype=np.int64)
    with torch.inference_mode():
        for i in range(n_tokens):
            pos = prompt_len + i
            last_np = last.detach().float().cpu().numpy().astype(np.float64, copy=False)

            closing = closing_genes(pos)
            if closing:
                # Codon-closing step: gate/reweight on the model's TRUE, untruncated
                # softmax over the 4 bases (top_k=None), not the top-4-truncated
                # distribution -- otherwise a base that's correct-but-not-in-the-
                # model's-own-top-4 (e.g. the real stop codon at a gene's end,
                # which the model essentially never ranks there unprompted) is
                # already zero before the gate ever runs, and forcing it is a
                # no-op. At non-closing steps this doesn't apply -- those stay on
                # the standard top_k=4 native sampling convention below.
                n_closing_steps += 1
                full_probs = apply_transform(last_np, temperature=1.0, top_k=None)
                seq_str = "".join(seq_chars)
                weights = candidate_weights(seq_str, pos, closing, gene_aa)
                base_probs = {b: full_probs[byte_id[b]] for b in BASES}
                any_rejected = False
                for b in BASES:
                    allowed, w = weights[b]
                    if not allowed:
                        if base_probs[b] > 0:
                            n_gate_rejections += 1
                            any_rejected = True
                        base_probs[b] = 0.0
                    elif w != 1.0:
                        base_probs[b] *= w
                total = sum(base_probs.values())
                if total > 0:
                    probs = np.zeros_like(full_probs)
                    for b in BASES:
                        probs[byte_id[b]] = base_probs[b] / total
                    if any_rejected:
                        for gi, s, e in closing:
                            if pos == e - 1:
                                n_forced_stop += 1
                else:
                    n_fallback += 1  # every one of the 4 bases zeroed (shouldn't happen: at most 3 of 4 can be stop codons) -- keep native top-4 probs this step
                    probs = apply_transform(last_np, temperature=1.0, top_k=4)
            else:
                probs = apply_transform(last_np, temperature=1.0, top_k=4)

            tok = sample_categorical(probs, rng)
            emitted[i] = tok
            ch = chr(tok) if 0 <= tok < 256 else "N"
            seq_chars.append(ch)

            if closing:
                advance_gene_aa("".join(seq_chars), pos, closing, gene_aa)

            if i + 1 < n_tokens:
                x = torch.tensor([[tok]], dtype=torch.long, device=device)
                step_logits = step_forward_reference(model, x, ip)
                last = step_logits[0, -1]
    torch.cuda.synchronize()
    guided_s = time.perf_counter() - t0
    es.close()

    from evspark.api import ids_to_text
    guided_seq = ids_to_text(emitted)

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
        "guided_s": round(guided_s, 2),
        "guided_toks_per_s": round(n_tokens / guided_s, 1) if guided_s > 0 else None,
        "n_closing_steps": n_closing_steps,
        "n_forced_stop": n_forced_stop,
        "n_gate_rejections": n_gate_rejections,
        "n_fallback": n_fallback,
        "est_usd_guided": round(guided_s / 3600 * price, 5),
        "guided_seq": guided_seq,
        "rounds": [],
    }


@app.local_entrypoint()
def kmer_guided_fulllength(model: str = "evo2_7b", ckpt: str = "L27_g12_150M_s1",
                            prompt_len: int = 64, n_tokens: int = 0,
                            aa_table: str = str(DEFAULT_AA_TABLE),
                            seeds: str = "0,1,2"):
    import bio_scoring
    import genome_scoring

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    ref_name, genome = read_fasta_records(str(PHIX174_FASTA))[0]
    genome_len = len(genome)
    tokens = n_tokens if n_tokens > 0 else genome_len - prompt_len
    prompt = genome[:prompt_len]
    aa_table_bytes = Path(aa_table).read_bytes()
    baseline = _load_baseline()
    print(f"phiX174 guided-decode run: {ref_name}, genome_len={genome_len}, "
          f"prompt_len={prompt_len}, n_tokens={tokens}, aa_table={aa_table}, "
          f"baseline={'loaded (' + str(len(baseline)) + ' entries)' if baseline else 'NOT FOUND'}")

    results = []
    for seed in seed_list:
        name = f"phix174_fulllength_guided_s{seed}"
        print(f"\n{name}: generating...")
        res = run_guided_prompt.remote(model, ckpt, name, "viral_coding_fulllength", prompt,
                                        tokens, seed, GPU, PRICE, aa_table_bytes)

        guided_seq = res["guided_seq"]
        real_seq = genome[prompt_len: prompt_len + tokens]
        has_real = len(real_seq) == tokens

        quality = {
            "guided": bio_scoring.score_sequence(guided_seq, organism_coding=True),
            "genome_guided": genome_scoring.score_genome(prompt + guided_seq, prompt_len=prompt_len),
        }
        if has_real:
            quality["real"] = bio_scoring.score_sequence(real_seq, organism_coding=True)
            quality["guided_vs_real"] = bio_scoring.compare_sequences(guided_seq, real_seq, organism_coding=True)
            quality["genome_real"] = genome_scoring.score_genome(genome, prompt_len=0)

        base = baseline.get(f"phix174_fulllength_s{seed}")
        if base:
            native_seq = base["native_seq"]
            quality["native_baseline"] = base["quality"]["native"]
            quality["guided_vs_native"] = bio_scoring.compare_sequences(guided_seq, native_seq, organism_coding=True)
            quality["genome_native_baseline"] = base["quality"]["genome_native"]
        else:
            print(f"  warning: no baseline entry for seed {seed} in {BASELINE_PATH.name}")

        res["quality"] = quality
        results.append(res)

        gk = quality["genome_guided"]
        print(f"  guided {res['guided_toks_per_s']} tok/s  "
              f"closing_steps={res['n_closing_steps']} forced_stop={res['n_forced_stop']}/8 "
              f"gate_rejections={res['n_gate_rejections']} fallback={res['n_fallback']}")
        print(f"  orf_closure: guided {gk['orf_closure']['n_closed']}/{gk['orf_closure']['n_scored']}"
              + (f"  native {quality['genome_native_baseline']['orf_closure']['n_closed']}/"
                 f"{quality['genome_native_baseline']['orf_closure']['n_scored']}" if base else ""))
        real_q = quality.get("real", {})
        print(f"  pyrodigal: guided n_genes={quality['guided'].get('n_genes')} "
              f"coding_density={quality['guided'].get('coding_density')}  "
              + (f"native n_genes={quality['native_baseline'].get('n_genes')} "
                 f"coding_density={quality['native_baseline'].get('coding_density')}  " if base else "")
              + f"real n_genes={real_q.get('n_genes')} coding_density={real_q.get('coding_density')}")

    out_path = f"kmer_fulllength_bench_{GPU}_guided.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model, "ckpt": ckpt, "prompt_len": prompt_len, "n_tokens": tokens,
            "seeds": seed_list, "genome_len": genome_len, "gpu": GPU, "price_per_hr": PRICE,
            "aa_table": str(aa_table), "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")
