"""Constrained nucleotide sampling for block generation: force a gene's
terminal stop codon to land where its real coordinates say it belongs,
instead of generating freely and hoping.

Why (2026-11-15 pilot, 17 rounds over 3 genomes, evo2_7b, c=3):

    rounds with a clean candidate                   0 / 17
    rounds missing the terminal stop               16 / 17
    rounds whose ONLY defect was the terminal stop  9 / 17
    rounds where PMI decided the pick               2 / 17

Nine rounds produced a completely stop-free ORF across the whole block in
the gene's real frame and still scored as broken, purely because the stop
codon did not land on the exact base. An unconstrained sampler has no reason
to place it there: it is one specific codon at one specific offset, and the
model is not told the block's length means anything. So the hard gate was
never measuring generation quality, it was measuring an accident.

Forcing that one codon converts those 9 rounds to clean outright and removes
one violation from the rest. Clean candidates then actually exist, which is
what lets PMI -- the thing the k-mer arm is supposed to be testing -- do the
selecting instead of violation count (it decided only 2 of 17 rounds).

What this deliberately does NOT do: suppress PREMATURE in-frame stops. It
would be easy (mask the third base of an in-frame codon whenever the first
two are TA or TG) and it would make every candidate pass the gate. It is the
wrong thing. The pilot's failing blocks carried premature stops at 3.8-4.5
per 100 codons against a chance rate of 4.7, two of them ABOVE chance --
meaning the model had slipped into a different frame and was generating
coding sequence there. Masking stops in a frame the model is not using
forces sequence it did not want and makes the output less realistic while
making the metric look better. Frame slippage is addressed where it starts,
by bounding block length (block_pipeline.split_long_blocks), not by hiding
its symptom here.

Mechanics: a stop codon is fully determined by its three bases and we always
know the frame, so the constraint is exact and local -- no lookahead:

    terminal codon, base 0:  allow {T}            (every stop starts with T)
    terminal codon, base 1:  allow {A, G}         (TA- or TG-)
    terminal codon, base 2:  allow {A, G} after TA; {A} after TG

The model still chooses among TAA/TAG/TGA by its own probabilities -- the
mask only removes the non-stop continuations.

Token ids are ASCII byte values (EvSpark's CharLevelTokenizer; see
evspark.api.ids_to_text), so 'A'=65, 'C'=67, 'G'=71, 'T'=84.

UNVALIDATED ON GPU as of 2026-11-15: the mask logic is unit-tested locally
(see _self_test) but this module has not yet run against a real model.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

A, C, G, T = 65, 67, 71, 84
NUCS = (A, C, G, T)
STOP_CODONS = ("TAA", "TAG", "TGA")


def terminal_stop_positions(
    block_start: int, block_len: int, genes: Sequence[str],
    gene_coords: Dict[str, Tuple[int, int]],
) -> Dict[int, int]:
    """{offset_into_block: index_within_codon} for every base of every
    terminal stop codon that belongs inside this block.

    A gene whose real end lies past `block_start + block_len` (a sub-block
    from split_long_blocks) contributes nothing: its stop codon is not in
    this block and must not be forced here."""
    out: Dict[int, int] = {}
    block_end = block_start + block_len
    for g in genes:
        gs, ge = gene_coords[g]
        if ge > block_end or ge - 3 < block_start:
            continue
        for j in range(3):
            pos = (ge - 3 + j) - block_start
            if 0 <= pos < block_len:
                out[pos] = j
    return out


def allowed_tokens(
    i: int, emitted: Sequence[int], stop_pos: Dict[int, int]
) -> Optional[Set[int]]:
    """Token ids permitted at offset `i`, or None for unconstrained.

    `emitted` holds the ids produced so far for this block, so the third base
    of a forced stop codon can see which of TA-/TG- it is completing."""
    j = stop_pos.get(i)
    if j is None:
        return None
    if j == 0:
        return {T}
    if j == 1:
        return {A, G}
    # j == 2: complete whichever stop the first two bases started
    if len(emitted) >= 2 and emitted[-2] == T:
        if emitted[-1] == A:
            return {A, G}       # TAA / TAG
        if emitted[-1] == G:
            return {A}          # TGA
    # The first two bases were not forced as expected (only reachable if a
    # caller passed an inconsistent stop_pos map) -- fall back to any stop
    # completion rather than silently emitting a non-stop codon.
    return {A, G}


def apply_mask(probs: np.ndarray, allowed: Optional[Set[int]]) -> np.ndarray:
    """Zero every token outside `allowed` and renormalize. If the model put no
    mass on any allowed token (top_k can exclude all of them), fall back to a
    uniform choice among the allowed set rather than returning all-zeros,
    which would make sampling undefined."""
    if allowed is None:
        return probs
    masked = np.zeros_like(probs)
    idx = [t for t in allowed if t < len(probs)]
    masked[idx] = probs[idx]
    total = masked.sum()
    if total <= 0:
        masked[idx] = 1.0 / len(idx)
        return masked
    return masked / total


def generate_block_constrained(
    model,
    prompt_ids,
    n_tokens: int,
    rng: np.random.Generator,
    stop_pos: Dict[int, int],
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = 4,
) -> np.ndarray:
    """Mirror of evspark.specdec.block.loop.native_sample_reference with a
    per-position mask applied between apply_transform and sample_categorical.

    Written here rather than patched into EvSpark so the cloned third-party
    repo stays untouched and reproducible (it is gitignored -- see
    .gitignore). The sampling loop already does a CPU round-trip per token,
    so the mask costs nothing measurable."""
    import torch
    from evspark.specdec.block.loop import (
        _as_prompt_ids, prefill, step_forward_reference,
    )
    from evspark.specdec.transforms import apply_transform
    from evspark.specdec.verifier import sample_categorical  # NOT in transforms -- checked

    prompt_ids = _as_prompt_ids(prompt_ids)
    if n_tokens <= 0:
        return np.empty(0, dtype=np.int64)
    device = prompt_ids.device
    L = int(prompt_ids.shape[1])
    ip = model.initialize_inference_params(max_seqlen=L + n_tokens + 2)
    emitted = np.empty(n_tokens, dtype=np.int64)
    with torch.inference_mode():
        logits = prefill(model, prompt_ids, ip)
        last = logits[0, -1]
        for i in range(n_tokens):
            last_np = last.detach().float().cpu().numpy().astype(np.float64, copy=False)
            probs = apply_transform(last_np, temperature=temperature, top_k=top_k)
            probs = apply_mask(probs, allowed_tokens(i, emitted[:i], stop_pos))
            tok = sample_categorical(probs, rng)
            emitted[i] = tok
            if i + 1 < n_tokens:
                x = torch.tensor([[tok]], dtype=torch.long, device=device)
                last = step_forward_reference(model, x, ip)[0, -1]
    return emitted


def _self_test() -> None:
    """Mask logic only -- no model, no GPU."""
    coords = {"g": (100, 400)}
    # Block covering the gene's end: the last 3 bases (397,398,399) are forced.
    sp = terminal_stop_positions(100, 300, ["g"], coords)
    assert sp == {297: 0, 298: 1, 299: 2}, sp
    # Sub-block NOT covering the gene's end: nothing forced.
    assert terminal_stop_positions(100, 150, ["g"], coords) == {}
    # Sub-block that starts after the gene's stop codon began: nothing forced.
    assert terminal_stop_positions(398, 50, ["g"], coords) == {}

    assert allowed_tokens(0, [], sp) is None
    assert allowed_tokens(297, [], sp) == {T}
    assert allowed_tokens(298, [T], sp) == {A, G}
    assert allowed_tokens(299, [T, A], sp) == {A, G}
    assert allowed_tokens(299, [T, G], sp) == {A}

    # Every reachable forced path spells a real stop codon, and nothing else.
    letters = {A: "A", C: "C", G: "G", T: "T"}
    spelled = set()
    for b1 in allowed_tokens(297, [], sp):
        for b2 in allowed_tokens(298, [b1], sp):
            for b3 in allowed_tokens(299, [b1, b2], sp):
                spelled.add(letters[b1] + letters[b2] + letters[b3])
    assert spelled == set(STOP_CODONS), spelled

    probs = np.zeros(256)
    probs[A], probs[C], probs[G], probs[T] = 0.1, 0.4, 0.2, 0.3
    m = apply_mask(probs, {T})
    assert abs(m[T] - 1.0) < 1e-12 and m[C] == 0.0
    m = apply_mask(probs, {A, G})
    assert abs(m[A] - 1 / 3) < 1e-12 and abs(m[G] - 2 / 3) < 1e-12
    # No mass on any allowed token -> uniform over them, never all-zero.
    zero = np.zeros(256)
    zero[C] = 1.0
    m = apply_mask(zero, {A, G})
    assert abs(m[A] - 0.5) < 1e-12 and abs(m[G] - 0.5) < 1e-12
    assert abs(apply_mask(probs, None).sum() - 1.0) < 1e-12

    print("constrained_decode self-test passed (mask logic only; GPU path unvalidated)")


if __name__ == "__main__":
    _self_test()
