"""Experiment 3: full-length (whole-replicon) phiX174 generation, phiX174 ONLY.

Earlier sweeps (speculative_benchmark.py) tested 5 organisms at short (~1kb)
windows to compare regions/speedup across species -- that question is answered.
This script asks a different one: does Evo2 (via EvSpark's speculative path,
still checked losslessly against native) hold up at whole-genome scale on a
single organism, where genome-level structure (gene closure, GC-skew origin
signal, terminator presence) becomes checkable at all? Single organism by
design, not an oversight -- phiX174 is the only genome in this repo small
enough (5386bp) to generate end-to-end in one shot with real annotation
on hand (ncbi_dataset_extracted/.../genomic.gbff, see genome_scoring.py).

Not an attempt to reproduce the Arc Institute phage-design paper (biorxiv
2025.09.12.675911) -- no fine-tuning, no design/diversification loop, no
tropism filtering. This evaluates base Evo2 (via EvSpark, unmodified target
model) at long-form generation, reusing that paper's choice of organism and
borrowing its idea of checking genome-level structure, not its pipeline.

Reuses speculative_benchmark.py's Modal app/image/run_prompt unchanged (same
GPU call, same EvSpark checkpoint, same native-vs-speculative losslessness
check) -- only difference is n_tokens is large enough to reach the end of the
genome from a short prompt, and scoring adds genome_scoring.py's whole-replicon
checks on top of bio_scoring.py's existing per-window metrics.

Prompt is genome[0:prompt_len] -- starting at position 0 (not an arbitrary
offset) matters: gene A/A*/B all span the circular origin (segments wrap
through position 0), so starting there means those three origin-spanning
genes always have at least their closing stop codon sitting in the prompt --
unavoidable for a linear one-shot continuation seeded near the origin, not
fixable by shrinking prompt_len further. orf_closure_check skips them (and
any other gene with any base in [0, prompt_len)) rather than crediting
sequence the model never generated. prompt_len=64 (not longer) keeps that loss as small as possible -- EvSpark's
own clean_prompt() hard-rejects prompts under 64bp ("prompt too short (<64
bp); give the model some context"), so 64 is the floor, not a free choice.
At 64bp, gene C (50-221bp) also straddles the boundary and gets excluded
alongside A/A*/B (4 of 11 lost); gene D (132-393bp) starts after position 64
and is fully scored. Any prompt_len above 64 would needlessly give away more.

Run:  BENCH_GPU=L40S GPU_PRICE_PER_HR=1.95 modal run fulllength_benchmark.py
Options: --prompt-len 64 --n-tokens 0 --seed 0 --model evo2_7b --ckpt L27_g12_150M_s1
         (--n-tokens 0 means "generate exactly to the end of the real genome",
         i.e. 5386 - prompt_len; pass a larger value to overshoot past the
         real genome length and test continuation past a natural stop point)
Writes: fulllength_bench_<GPU>.json
"""
import json
import os
from pathlib import Path

from speculative_benchmark import GPU, PRICE, app, read_fasta_records, run_prompt

PHIX174_FASTA = Path(__file__).parent / "phix174.fasta"
SEEDS = [0, 1, 2]


@app.local_entrypoint()
def fulllength(model: str = "evo2_7b", ckpt: str = "L27_g12_150M_s1",
          prompt_len: int = 64, n_tokens: int = 0, gamma: int = 0,
          seeds: str = "0,1,2"):
    import bio_scoring
    import genome_scoring

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    ref_name, genome = read_fasta_records(str(PHIX174_FASTA))[0]
    genome_len = len(genome)
    tokens = n_tokens if n_tokens > 0 else genome_len - prompt_len
    prompt = genome[:prompt_len]
    print(f"phiX174 full-length run: {ref_name}, genome_len={genome_len}, "
          f"prompt_len={prompt_len}, n_tokens={tokens} "
          f"(generated end = prompt_len + n_tokens = {prompt_len + tokens}, "
          f"genome_len = {genome_len})")

    results = []
    for seed in seed_list:
        name = f"phix174_fulllength_s{seed}"
        print(f"\n{name}: generating...")
        res = run_prompt.remote(model, ckpt, name, "viral_coding_fulllength", prompt,
                                 tokens, gamma, seed, GPU, PRICE)

        native_seq, spec_seq = res["native_seq"], res["speculative_seq"]
        real_seq = genome[prompt_len: prompt_len + tokens]
        has_real = len(real_seq) == tokens

        quality = {
            "native": bio_scoring.score_sequence(native_seq, organism_coding=True),
            "speculative": bio_scoring.score_sequence(spec_seq, organism_coding=True),
            "native_vs_speculative": bio_scoring.compare_sequences(native_seq, spec_seq, organism_coding=True),
            "genome_native": genome_scoring.score_genome(prompt + native_seq, prompt_len=prompt_len),
            "genome_speculative": genome_scoring.score_genome(prompt + spec_seq, prompt_len=prompt_len),
        }
        if has_real:
            quality["real"] = bio_scoring.score_sequence(real_seq, organism_coding=True)
            quality["native_vs_real"] = bio_scoring.compare_sequences(native_seq, real_seq, organism_coding=True)
            quality["speculative_vs_real"] = bio_scoring.compare_sequences(spec_seq, real_seq, organism_coding=True)
            quality["genome_real"] = genome_scoring.score_genome(genome, prompt_len=0)
        res["quality"] = quality
        results.append(res)

        gn, gs_ = quality["genome_native"], quality["genome_speculative"]
        print(f"  native {res['native_toks_per_s']} tok/s  speculative {res['speculative_toks_per_s']} tok/s  "
              f"speedup {res['speedup']}x  acceptance={res['acceptance_rate']}")
        print(f"  orf_closure: native {gn['orf_closure']['n_closed']}/{gn['orf_closure']['n_scored']}  "
              f"speculative {gs_['orf_closure']['n_closed']}/{gs_['orf_closure']['n_scored']}")
        print(f"  gc_skew sign changes: native {gn['gc_skew']['n_sign_changes']}  "
              f"speculative {gs_['gc_skew']['n_sign_changes']}  "
              f"(real {quality.get('genome_real', {}).get('gc_skew', {}).get('n_sign_changes')})")
        print(f"  terminator hits: native {gn['terminators']['n_hits']}  speculative {gs_['terminators']['n_hits']}")
        # Independent Pyrodigal gene-calling on the generated region itself (not checked
        # against real coordinates) -- a looser complement to orf_closure's strict exact-
        # coordinate test: does the generated sequence look gene-structurally-plausible
        # ANYWHERE, regardless of whether it lines up with phiX174's real gene boundaries.
        real_pyro = quality.get("real", {})
        print(f"  pyrodigal (independent gene-calling on generated region): "
              f"native n_genes={quality['native'].get('n_genes')} "
              f"coding_density={quality['native'].get('coding_density')}  "
              f"speculative n_genes={quality['speculative'].get('n_genes')} "
              f"coding_density={quality['speculative'].get('coding_density')}  "
              f"real n_genes={real_pyro.get('n_genes')} coding_density={real_pyro.get('coding_density')}")

    out_path = f"fulllength_bench_{GPU}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model, "ckpt": ckpt, "prompt_len": prompt_len, "n_tokens": tokens,
            "seeds": seed_list, "genome_len": genome_len, "gpu": GPU, "price_per_hr": PRICE,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")
