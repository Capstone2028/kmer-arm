"""Experiment 1: native vs. EvSpark speculative decoding, instrumented per round.

Uses the real EvSpark public API (github.com/dhnihaoya/EvSpark, cloned locally at
./EvSpark) rather than a guessed interface: evspark.EvSpark.load/.generate_native
for the summary path, and evspark.specdec.block.loop.speculative_generate
directly (the same call EvSpark.generate() makes internally) so we keep the full
per-round RoundLog instead of just the aggregated mean_tau/n_rounds.

Two defaults inherited from EvSpark itself, different from the rest of this repo:
  - model_name="evo2_7b" (not "evo2_7b_base" as in evo2_modal_benchmark.py /
    baseline_generations.py) -- EvSpark.load()'s own default; flagged here in
    case it turns out to be the wrong variant once this actually runs.
  - default checkpoint "L27_g12_150M_s1" (gamma=12), not the gamma=7 reference
    model discussed in the paper's prose -- this is the checkpoint EvSpark's own
    checkpoints.py lists as the "flagship" default.

UNTESTED on Modal/GPU. The API surface is taken directly from the cloned repo's
evspark/api.py and evspark/specdec/block/loop.py, so this should be close, but
hasn't been run against a real GPU yet.

Needs the five genome prompt files already in this folder (phix174.fasta,
ecoli_k12.fasta, bsubtilis.fasta, lambda_phage.fasta, chr21_windows.fasta).

Run:  BENCH_GPU=L40S GPU_PRICE_PER_HR=1.95 modal run speculative_benchmark.py
      (PowerShell: $env:BENCH_GPU="L40S"; $env:GPU_PRICE_PER_HR="1.95")
Options: --n-tokens 1024 --gamma 0 --seed 0 --model evo2_7b --ckpt L27_g12_150M_s1
         (--gamma 0 means "use the checkpoint's trained gamma")
Writes:  speculative_bench_<GPU>.json (per-prompt summary: tau_bar, acceptance
         rate, native/speculative wall time, speedup) and
         speculative_rounds.csv (one row per speculative round: k, replay_len,
         and the t_draft/t_block/t_verify/t_slice timing breakdown EvSpark's
         RoundLog already tracks).
"""
import csv
import json
import os
import time
from pathlib import Path

import modal

GPU = os.environ.get("BENCH_GPU", "L40S")
PRICE = float(os.environ.get("GPU_PRICE_PER_HR", "1.95"))
EVSPARK_DIR = Path(__file__).parent / "EvSpark"

# One genome per EvSpark region category (Table 2). Windows for the four
# "coding" genomes come from coding_windows.json (run pick_coding_windows.py
# to regenerate) -- up to 3 non-overlapping Pyrodigal-verified coding windows
# per genome, not an arbitrary genome[:1000] slice. Without that check,
# "bacterial_coding"/"viral_coding" was just a label we hadn't verified:
# phiX174's old genome[:1000] window turned out to be coding_density=0.575
# (barely above the genome's own average), not actually representative.
#
# chr21 windows come from chr21_windows.json (run pick_human_windows.py to
# regenerate) -- up to 3 non-overlapping UCSC-gene-gap-verified intergenic
# windows, picked from the largest gaps between ncbiRefSeq transcripts on
# chr21 after excluding the centromere and assembly-gap tracks (Pyrodigal is
# a prokaryotic/phage gene-finder, not applicable to human sequence, so gene
# gaps come from UCSC annotation instead of a gene-finder run). This used to
# be a single hardcoded window (chr21_window.fasta) -- n=1 for the whole
# human_intergenic region, unlike every other genome here.
#
# Still one organism per region (not one window = one species) -- multiple
# windows from the same genome share that genome's codon/GC bias, same
# pseudo-replication caveat as always; see the chat writeup.
GENOMES = {
    "phix174": {"region": "viral_coding", "fasta": "phix174.fasta"},
    "ecoli_k12": {"region": "bacterial_coding", "fasta": "ecoli_k12.fasta"},
    "bsubtilis": {"region": "bacterial_coding", "fasta": "bsubtilis.fasta"},
    "lambda_phage": {"region": "viral_coding", "fasta": "lambda_phage.fasta"},
}
CHR21 = {"name": "chr21", "region": "human_intergenic", "fasta": "chr21_windows.fasta"}
SEEDS = [0, 1]


def build_prompts(seeds: list[int]) -> list[dict]:
    """GENOMES x their verified windows (coding_windows.json) x seeds, plus
    chr21's verified windows (chr21_windows.json) x seeds. One dict per
    (genome, window, seed) run."""
    prompts = []
    windows_path = Path(__file__).parent / "coding_windows.json"
    coding_windows = json.loads(windows_path.read_text()) if windows_path.exists() else {}
    for genome, meta in GENOMES.items():
        windows = coding_windows.get(genome, {}).get("windows", [])
        if not windows:
            print(f"warning: no entry for {genome} in coding_windows.json "
                  f"(run pick_coding_windows.py) -- skipping")
            continue
        for w_idx, w in enumerate(windows):
            for seed in seeds:
                prompts.append({
                    "name": f"{genome}_w{w_idx}_s{seed}", "region": meta["region"], "fasta": meta["fasta"],
                    "prompt_start": w["window_start"], "prompt_end": w["window_end"], "seed": seed,
                })
    chr21_windows_path = Path(__file__).parent / "chr21_windows.json"
    chr21_windows = json.loads(chr21_windows_path.read_text()) if chr21_windows_path.exists() else {}
    windows = chr21_windows.get(CHR21["name"], {}).get("windows", [])
    if not windows:
        print(f"warning: no entry for {CHR21['name']} in chr21_windows.json "
              f"(run pick_human_windows.py) -- skipping")
    for w_idx, w in enumerate(windows):
        for seed in seeds:
            prompts.append({
                "name": f"{CHR21['name']}_w{w_idx}_s{seed}", "region": CHR21["region"], "fasta": CHR21["fasta"],
                # chr21_windows.fasta holds one record per window, in the same order as
                # chr21_windows.json, so the window's own length is its slice within that record.
                "prompt_start": 0, "prompt_end": w["window_end"] - w["window_start"], "seed": seed,
                "fasta_record": w_idx,
            })
    return prompts

app = modal.App("evo2-speculative")
cache = modal.Volume.from_name("evo2-hf-cache", create_if_missing=True)

# Same base as evo2_modal_benchmark.py / baseline_generations.py (Modal reuses
# the cached layers), with the local EvSpark clone baked in and pip-installed
# per its own pyproject.toml (which pulls evo2 as a dependency already).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("packaging", "wheel", "setuptools")
    .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu128")
    .run_commands("pip install flash-attn==2.8.0.post2 --no-build-isolation")
    .add_local_dir(str(EVSPARK_DIR), remote_path="/opt/evspark", copy=True)
    .run_commands("cd /opt/evspark && pip install -e .")
    .env({"HF_HOME": "/cache/hf", "EVSPARK_CKPT_DIR": "/cache/evspark_ckpts"})
)


@app.function(image=image, gpu=GPU, volumes={"/cache": cache}, timeout=3 * 60 * 60)
def run_prompt(model_name: str, ckpt: str, name: str, region: str, prompt: str,
                n_tokens: int, gamma: int, seed: int, gpu: str, price: float) -> dict:
    import numpy as np
    import torch
    from evspark import EvSpark
    from evspark.api import clean_prompt, ids_to_text
    from evspark.specdec.block.loop import speculative_generate

    t0 = time.perf_counter()
    es = EvSpark.load(ckpt=ckpt, model_name=model_name, device="cuda:0")
    load_s = time.perf_counter() - t0
    cache.commit()

    g = int(gamma) if gamma else es.gamma  # gamma=0 -> use the checkpoint's trained gamma

    seq = clean_prompt(prompt)
    ids = torch.tensor(
        np.asarray(es.tokenizer.tokenize(seq), dtype=np.int64)[None, :], device=es.device,
    )

    # --- warm-up, untimed: first call into each code path triggers CUDA kernel
    # JIT compilation / autotuning (seen as a ~7x t_block drop between round 0 and
    # round 1 in the first --quick run). Pay that cost here, on both paths, before
    # either timed measurement below, so neither arm's number is a cold-start artifact.
    WARMUP_TOKENS = 32
    es.generate_native(prompt, n_tokens=WARMUP_TOKENS, greedy=False, seed=seed,
                        temperature=1.0, top_k=4)
    speculative_generate(
        es.model, es._nd, ids, WARMUP_TOKENS, g, greedy=False,
        rng=np.random.default_rng(seed), temperature=1.0, top_k=4, rollback="slice",
    )
    torch.cuda.synchronize()

    # --- native reference, via the public wrapper (no per-round structure to capture) ---
    nat = es.generate_native(prompt, n_tokens=n_tokens, greedy=False, seed=seed,
                              temperature=1.0, top_k=4)

    # --- speculative, calling the lower-level loop directly to keep rounds_log ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spec = speculative_generate(
        es.model, es._nd, ids, int(n_tokens), g, greedy=False,
        rng=np.random.default_rng(seed), temperature=1.0, top_k=4, rollback="slice",
    )
    torch.cuda.synchronize()
    spec_s = time.perf_counter() - t0
    es.close()

    rounds = [{
        "round": i, "gamma": g, "k_accepted": int(r.k), "replay_len": int(r.replay_len),
        "t_draft": r.t_draft, "t_block": r.t_block, "t_verify": r.t_verify,
        "t_restore_replay": r.t_restore_replay, "t_snapshot": r.t_snapshot, "t_slice": r.t_slice,
        "n_emitted": r.n_emitted, "fallback": r.fallback, "from_residual": r.from_residual,
    } for i, r in enumerate(spec.rounds_log)]

    k_vals = [r["k_accepted"] for r in rounds]
    tau_bar = (sum(k_vals) / len(k_vals) + 1.0) if k_vals else 0.0  # +1: accepted draft tokens + the bonus/anchor token
    acceptance_rate = sum(k / g for k in k_vals) / len(k_vals) if k_vals else 0.0

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
        "gamma": g,
        "seed": seed,
        "native_s": round(nat.wall_s, 2),
        "native_toks_per_s": round(nat.tok_s, 1),
        "speculative_s": round(spec_s, 2),
        "speculative_toks_per_s": round(n_tokens / spec_s, 1) if spec_s > 0 else None,
        "speedup": round(nat.wall_s / spec_s, 3) if spec_s > 0 else None,
        "n_rounds": len(rounds),
        "tau_bar": round(tau_bar, 3),
        "acceptance_rate": round(acceptance_rate, 3),
        "est_usd_native": round(nat.wall_s / 3600 * price, 5),
        "est_usd_speculative": round(spec_s / 3600 * price, 5),
        "native_seq": nat.text,
        "speculative_seq": ids_to_text(spec.emitted_ids),
        "rounds": rounds,
    }


def read_fasta_records(path):
    records = []
    name, buf = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf).upper()))
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name is not None:
        records.append((name, "".join(buf).upper()))
    return records


def read_first_fasta(path):
    return read_fasta_records(path)[0]


@app.local_entrypoint()
def main(model: str = "evo2_7b", ckpt: str = "L27_g12_150M_s1",
          n_tokens: int = 1024, gamma: int = 0, quick: bool = False, only: str = ""):
    import bio_scoring  # local CPU import only -- never runs inside the Modal container

    seeds = SEEDS[:1] if quick else SEEDS
    prompts = build_prompts(seeds)
    if only:
        prompts = [p for p in prompts if p["name"].startswith(only)]
    if quick:
        prompts = prompts[:1]
        n_tokens = min(n_tokens, 128)
        print(f"--quick: one prompt ({prompts[0]['name']}), n_tokens={n_tokens}")

    results = []
    for p in prompts:
        if not os.path.exists(p["fasta"]):
            print(f"skip {p['name']}: {p['fasta']} not found")
            continue
        ref_name, genome = read_fasta_records(p["fasta"])[p.get("fasta_record", 0)]
        prompt = genome[p["prompt_start"]: p["prompt_end"]]
        print(f"{p['name']} ({p['region']}, seed={p['seed']}): prompt from {ref_name} "
              f"[{p['prompt_start']}:{p['prompt_end']}], {len(prompt)} bp")
        res = run_prompt.remote(model, ckpt, p["name"], p["region"], prompt,
                                 n_tokens, gamma, p["seed"], GPU, PRICE)

        # --- biological quality scoring, CPU-local, on the sequences the GPU
        # call just returned. Pairs every speed number above with a quality
        # number: native vs. speculative should match within noise (validates
        # losslessness at the statistic level, since sampling means the raw
        # token streams diverge even when both are correct draws), and both
        # are compared against the real genomic continuation as a sanity floor.
        organism_coding = p["region"] != "human_intergenic"  # pyrodigal is prokaryotic/phage-only
        native_seq, spec_seq = res["native_seq"], res["speculative_seq"]
        real_seq = genome[p["prompt_end"]: p["prompt_end"] + n_tokens]
        if len(real_seq) < n_tokens:
            real_seq = None  # window ran off the end of this record; skip the real-continuation comparison

        quality = {
            "native": bio_scoring.score_sequence(native_seq, organism_coding),
            "speculative": bio_scoring.score_sequence(spec_seq, organism_coding),
            "native_vs_speculative": bio_scoring.compare_sequences(native_seq, spec_seq, organism_coding),
        }
        if real_seq:
            quality["real"] = bio_scoring.score_sequence(real_seq, organism_coding)
            quality["native_vs_real"] = bio_scoring.compare_sequences(native_seq, real_seq, organism_coding)
            quality["speculative_vs_real"] = bio_scoring.compare_sequences(spec_seq, real_seq, organism_coding)
        res["quality"] = quality
        results.append(res)

        nvs = quality["native_vs_speculative"]
        print(f"  native {res['native_toks_per_s']} tok/s  speculative {res['speculative_toks_per_s']} tok/s  "
              f"speedup {res['speedup']}x  tau={res['tau_bar']}  acceptance={res['acceptance_rate']}")
        print(f"  quality: native gc={quality['native']['gc']} spec gc={quality['speculative']['gc']}  "
              f"coding_density native={quality['native'].get('coding_density')} "
              f"spec={quality['speculative'].get('coding_density')}  "
              f"kmer_js(native,spec)={nvs['kmer_js_divergence']}  codon_js={nvs['codon_js_divergence']}")

    out_path = f"speculative_bench_{GPU}{'_' + only if only else ''}.json"
    with open(out_path, "w") as f:
        json.dump({"model": model, "ckpt": ckpt, "n_tokens": n_tokens, "gamma": gamma, "seeds": seeds,
                    "gpu": GPU, "price_per_hr": PRICE, "results": results}, f, indent=2)
    print(f"\nWrote {out_path}")

    rounds_path = f"speculative_rounds{'_' + only if only else ''}.csv"
    cols = ["prompt", "region", "round", "gamma", "k_accepted", "replay_len",
            "t_draft", "t_block", "t_verify", "t_restore_replay", "t_snapshot", "t_slice",
            "n_emitted", "fallback", "from_residual"]
    with open(rounds_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for res in results:
            for r in res["rounds"]:
                w.writerow([res["name"], res["region"]] + [r[c] for c in cols[2:]])
    print(f"Wrote {rounds_path}")

    quality_path = f"speculative_quality_{GPU}{'_' + only if only else ''}.csv"
    q_cols = ["prompt", "region", "speedup", "acceptance_rate",
              "gc_native", "gc_speculative", "gc_real",
              "complexity_native", "complexity_speculative", "complexity_real",
              "coding_density_native", "coding_density_speculative", "coding_density_real",
              "min_frame_stop_native", "min_frame_stop_speculative", "min_frame_stop_real",
              "kmer_js_native_vs_speculative", "codon_js_native_vs_speculative",
              "kmer_js_native_vs_real", "kmer_js_speculative_vs_real"]
    with open(quality_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(q_cols)
        for res in results:
            q = res["quality"]
            real = q.get("real", {})
            row = {
                "prompt": res["name"], "region": res["region"],
                "speedup": res["speedup"], "acceptance_rate": res["acceptance_rate"],
                "gc_native": q["native"]["gc"], "gc_speculative": q["speculative"]["gc"],
                "gc_real": real.get("gc"),
                "complexity_native": q["native"]["complexity"], "complexity_speculative": q["speculative"]["complexity"],
                "complexity_real": real.get("complexity"),
                "coding_density_native": q["native"].get("coding_density"),
                "coding_density_speculative": q["speculative"].get("coding_density"),
                "coding_density_real": real.get("coding_density"),
                "min_frame_stop_native": q["native"]["min_frame"],
                "min_frame_stop_speculative": q["speculative"]["min_frame"],
                "min_frame_stop_real": real.get("min_frame"),
                "kmer_js_native_vs_speculative": q["native_vs_speculative"]["kmer_js_divergence"],
                "codon_js_native_vs_speculative": q["native_vs_speculative"]["codon_js_divergence"],
                "kmer_js_native_vs_real": q.get("native_vs_real", {}).get("kmer_js_divergence"),
                "kmer_js_speculative_vs_real": q.get("speculative_vs_real", {}).get("kmer_js_divergence"),
            }
            w.writerow([row[c] for c in q_cols])
    print(f"Wrote {quality_path}")
