#!/usr/bin/env python3
"""
HNET STEP 6A — CEBS-INSPIRED CONTEXTUAL ENERGY-BASED SLOTTING (TEXT) — IMPROVED

This is an improved version of Step 6 that fixes two major pathologies seen in the
previous run output:

(1) "Phantom entities" from inactive roles
    - Previously, roles that never appeared in a segment had near-zero signatures,
      which forced births each segment. That produced births≈15/segment, stability≈1,
      surprise≈0, and a huge registry dominated by junk entities.
    - Now: a role must be ACTIVE in a segment (count >= --min_role_count) to participate
      in binding. Inactive roles are left UNBOUND and do not spawn entities.

(2) Empty/degenerate role labels + punctuation role dominating
    - Now: role labels skip pure punctuation tokens by default and report both
      (a) global top tokens and (b) per-segment top tokens for sampled segments.

CEBS-inspired pieces reflected here:
  - Entity registry with birth/retrieve/retire.
  - Context-signature stability used as an attention/diagnostic signal.
  - Adaptive EMA update rate for canonical context (alpha depends on stability),
    matching CEBS' "adaptive streaming updates" idea.
  - Optional simple "message passing" over the role graph to smooth signatures,
    echoing CEBS' message passing context propagation (implemented as a cheap linear
    propagation, not learned).

This script is intended to remain:
  - single-file, runnable locally
  - fast (JAX-jitted role clustering; vectorized numpy binding)
  - verbose (prints extensive CLI diagnostics)
  - produces: JSON report + plots + a plaintext summary file

Usage:
  python hnet_step6a_cebs_contextual_energy_slotting_text_improved.py --input input.txt

Recommended:
  - If you see too few entities, lower --birth_thresh slightly (e.g. 0.65).
  - If you see too many births, raise --min_role_count or raise --birth_thresh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_N_CLUSTERS = 16
DEFAULT_N_SWEEPS = 40
DEFAULT_TOPK = 64
DEFAULT_SEG_LEN = 256
DEFAULT_N_SEGS = 32
DEFAULT_SEED = 0

# clustering dynamics (kept from step6)
LAMBDA_DIR = 0.8
GAMMA_SIZE = 1.0
UPDATE_PROB = 0.25

# registry dynamics
DEFAULT_BIRTH_THRESH = 0.72
DEFAULT_RETIRE_AFTER = 10

# adaptive alpha (canonical context update)
ALPHA_BASE = 0.90
STABILITY_THRESH = 0.65
ALPHA_BETA = 8.0

# relational surprise
DEFAULT_SURPRISE_THRESH = 0.18

# signature smoothing ("message passing" over roles)
DEFAULT_MSGPASS_BETA = 0.35  # 0 disables smoothing


# =============================================================================
# Utilities
# =============================================================================

def now() -> str:
    import datetime as _dt
    return _dt.datetime.now().isoformat(sep=" ", timespec="seconds")

_TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

def read_text(path: str, max_chars: int | None = None) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if max_chars is not None:
        txt = txt[:max_chars]
    return txt

def is_punct(tok: str) -> bool:
    # treat single-char non-alnum as punctuation-ish
    return (len(tok) <= 2) and (re.fullmatch(r"[^A-Za-z0-9]+", tok) is not None)

def build_vocab(tokens: List[str], min_count: int = 2) -> Tuple[Dict[str, int], List[str], np.ndarray]:
    from collections import Counter
    cnt = Counter(tokens)
    vocab = {"<UNK>": 0}
    id_to_tok = ["<UNK>"]
    for tok, c in cnt.most_common():
        if c < min_count:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(id_to_tok)
        id_to_tok.append(tok)
    ids = np.array([vocab.get(t, 0) for t in tokens], dtype=np.int32)
    return vocab, id_to_tok, ids

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def cosine_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Cosine similarity between each row of A (m,d) and each row of B (n,d).
    Returns (m,n).
    """
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return (An @ Bn.T).astype(np.float32)

def adaptive_alpha(stability: float) -> float:
    # alpha = alpha_base + (1-alpha_base)*sigmoid(beta*(stability-thresh))
    sig = 1.0 / (1.0 + math.exp(-ALPHA_BETA * (stability - STABILITY_THRESH)))
    return float(ALPHA_BASE + (1.0 - ALPHA_BASE) * sig)

def safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


# =============================================================================
# Directional co-occurrence (fast, numpy)
# =============================================================================

def build_directional_edges(ids: np.ndarray, max_dist: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build weighted directed edges i->j for pairs within max_dist.
    Weight decays with distance (1/sqrt(d)).
    """
    t0 = time.time()
    from collections import defaultdict
    acc = defaultdict(float)
    n = len(ids)
    for d in range(1, max_dist + 1):
        w = 1.0 / math.sqrt(d)
        src = ids[:-d]
        dst = ids[d:]
        pairs = (src.astype(np.int64) << 32) | dst.astype(np.int64)
        upairs, counts = np.unique(pairs, return_counts=True)
        contrib = counts.astype(np.float64) * w
        for p, c in zip(upairs.tolist(), contrib.tolist()):
            acc[p] += c

    E = len(acc)
    edges_i = np.empty(E, dtype=np.int32)
    edges_j = np.empty(E, dtype=np.int32)
    edges_w = np.empty(E, dtype=np.float32)
    for idx, (p, w) in enumerate(acc.items()):
        edges_i[idx] = np.int32(p >> 32)
        edges_j[idx] = np.int32(p & 0xFFFFFFFF)
        edges_w[idx] = np.float32(w)

    dt = time.time() - t0
    print("Building directional co-occurrence (fast)...")
    print(f"Cooc unique edges: {E:,}  (time {dt:.2f}s)")
    return edges_i, edges_j, edges_w


# =============================================================================
# Sparse assoc (top-k outgoing per token)
# =============================================================================

def build_sparse_assoc(vocab_size: int, edges_i: np.ndarray, edges_j: np.ndarray, edges_w: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each token i, keep top-k outgoing neighbors j with highest weight.
    """
    t0 = time.time()
    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(vocab_size)]
    for i, j, w in zip(edges_i.tolist(), edges_j.tolist(), edges_w.tolist()):
        buckets[i].append((j, float(w)))

    nbrs = np.zeros((vocab_size, topk), dtype=np.int32)
    wts = np.zeros((vocab_size, topk), dtype=np.float32)
    for i in range(vocab_size):
        if not buckets[i]:
            continue
        buckets[i].sort(key=lambda t: t[1], reverse=True)
        sel = buckets[i][:topk]
        for k, (j, w) in enumerate(sel):
            nbrs[i, k] = np.int32(j)
            wts[i, k] = np.float32(w)

    dt = time.time() - t0
    print(f"Sparse assoc built: topk={topk}  (time {dt:.2f}s)")
    return nbrs, wts


# =============================================================================
# Balanced + inertial clustering (roles) — JAX, with static n_clusters
# =============================================================================

def cluster_roles_jax(
    vocab_size: int,
    nbrs_np: np.ndarray,
    wts_np: np.ndarray,
    n_clusters: int,
    n_sweeps: int,
    seed: int,
    gamma_size: float,
    update_prob: float,
    lambda_dir: float,
) -> np.ndarray:
    """
    Returns role_of_word: (V,) int32
    """
    assert isinstance(n_clusters, int), "n_clusters must be a Python int (static)."
    C = n_clusters

    nbrs = jnp.array(nbrs_np, dtype=jnp.int32)   # (V,topk)
    wts  = jnp.array(wts_np,  dtype=jnp.float32) # (V,topk)

    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    state0 = jax.random.randint(k0, shape=(vocab_size,), minval=0, maxval=C, dtype=jnp.int32)

    def one_hot_state(state):
        return jnp.eye(C, dtype=jnp.float32)[state]  # (V,C)

    def energy_terms(state):
        S = one_hot_state(state)         # (V,C)
        Sn = S[nbrs]                     # (V,topk,C)
        assoc = jnp.einsum("vk,vkc->vc", wts, Sn)  # (V,C)
        counts = jnp.sum(S, axis=0)      # (C,)
        target = vocab_size / float(C)
        size_pen = gamma_size * (counts - target) / (target + 1e-6)  # (C,)
        E = -lambda_dir * assoc + size_pen[None, :]
        return E, counts

    @jax.jit
    def sweep(state, key):
        E, counts = energy_terms(state)
        key, kmask = jax.random.split(key, 2)
        mask = jax.random.bernoulli(kmask, p=update_prob, shape=(vocab_size,))
        new_state = jnp.argmin(E, axis=1).astype(jnp.int32)
        state = jnp.where(mask, new_state, state)
        return state, key, counts

    state = state0
    for s in range(n_sweeps):
        state, key, counts = sweep(state, key)
        if (s == 0) or ((s + 1) % 10 == 0) or (s == n_sweeps - 1):
            counts_np = np.array(counts, dtype=np.float32)
            nonempty = int(np.sum(counts_np > 0.5))
            print(f"  sweep {s+1}/{n_sweeps}  nonempty={nonempty}")

    return np.array(state, dtype=np.int32)


# =============================================================================
# Role-level couplings & segment signatures
# =============================================================================

def role_directional_coupling(role_of_word: np.ndarray, edges_i: np.ndarray, edges_j: np.ndarray, edges_w: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Aggregate token-level directed edges into role->role coupling matrix J_cc (normalized to [0,1]).
    """
    C = n_clusters
    ri = role_of_word[edges_i]
    rj = role_of_word[edges_j]
    J = np.zeros((C, C), dtype=np.float64)
    np.add.at(J, (ri, rj), edges_w.astype(np.float64))
    mx = float(J.max()) if J.size else 1.0
    if mx > 0:
        J /= mx
    return J.astype(np.float32)

def role_transition_signature(role_ids: np.ndarray, n_clusters: int, laplace: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    role_ids: (S,) ints in [0,C)
    Returns:
      sig: (C, 2C) per-role signature
      counts: (C,) per-role occurrence counts inside the segment

    Signature = [outgoing probs (C), incoming probs (C)] with light Laplace smoothing.
    """
    C = n_clusters
    counts = np.bincount(role_ids, minlength=C).astype(np.int32)
    x = role_ids[:-1]
    y = role_ids[1:]
    M = np.zeros((C, C), dtype=np.float32)
    np.add.at(M, (x, y), 1.0)

    # outgoing distribution per row
    out = (M + laplace) / (M.sum(axis=1, keepdims=True) + laplace * C + 1e-6)
    # incoming distribution per column
    inn = (M + laplace) / (M.sum(axis=0, keepdims=True) + laplace * C + 1e-6)

    sig = np.concatenate([out, inn.T], axis=1).astype(np.float32)  # (C,2C)
    return sig, counts

def smooth_signatures(sig: np.ndarray, J_cc: np.ndarray, beta: float) -> np.ndarray:
    """
    Cheap "message passing": mix each role's signature with neighbors via J_cc.

      sig_sm = sig + beta * (J_cc @ out_part, J_cc.T @ in_part)

    This is a heuristic echo of CEBS-style context propagation (not learned).
    """
    if beta <= 0:
        return sig
    C = J_cc.shape[0]
    out = sig[:, :C]
    inn = sig[:, C:]
    out_sm = out + beta * (J_cc @ out)
    inn_sm = inn + beta * (J_cc.T @ inn)
    # renormalize rows to keep distributions sane
    out_sm = out_sm / (out_sm.sum(axis=1, keepdims=True) + 1e-6)
    inn_sm = inn_sm / (inn_sm.sum(axis=1, keepdims=True) + 1e-6)
    return np.concatenate([out_sm, inn_sm], axis=1).astype(np.float32)


# =============================================================================
# Entity Registry
# =============================================================================

@dataclass
class Entity:
    eid: int
    canon: np.ndarray         # (2C,)
    last_seen: int
    confidence: float
    bound_role: int | None    # last role it was bound to (for interpretability)
    bindings: int             # how many total bindings (for ranking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--outdir", type=str, default="step6a_out")
    ap.add_argument("--max_chars", type=int, default=0, help="If >0, truncate input to this many chars")
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--max_dist", type=int, default=128)

    ap.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    ap.add_argument("--n_sweeps", type=int, default=DEFAULT_N_SWEEPS)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--seg_len", type=int, default=DEFAULT_SEG_LEN)
    ap.add_argument("--n_segs", type=int, default=DEFAULT_N_SEGS)

    # step6a improvements
    ap.add_argument("--min_role_count", type=int, default=6,
                    help="Role must appear at least this many times in a segment to be considered ACTIVE.")
    ap.add_argument("--birth_thresh", type=float, default=DEFAULT_BIRTH_THRESH)
    ap.add_argument("--retire_after", type=int, default=DEFAULT_RETIRE_AFTER)
    ap.add_argument("--surprise_thresh", type=float, default=DEFAULT_SURPRISE_THRESH)
    ap.add_argument("--msgpass_beta", type=float, default=DEFAULT_MSGPASS_BETA)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 80)
    print("HNET STEP 6A — CEBS CONTEXTUAL ENERGY SLOTTING (TEXT) — IMPROVED")
    print("=" * 80)

    txt = read_text(args.input, max_chars=(args.max_chars if args.max_chars > 0 else None))
    toks = tokenize(txt)
    print(f"Loaded: {args.input}")
    print(f"Raw tokens: {len(toks):,}")

    vocab, id_to_tok, ids = build_vocab(toks, min_count=args.min_count)
    V = len(id_to_tok)
    print(f"Vocab size (min_count={args.min_count}): {V:,}")
    in_vocab = int(np.sum(ids != 0))
    print(f"In-vocab token stream length: {in_vocab:,}")

    edges_i, edges_j, edges_w = build_directional_edges(ids, max_dist=args.max_dist)
    print(f"Edges arrays: E={len(edges_i):,}")

    nbrs, wts = build_sparse_assoc(V, edges_i, edges_j, edges_w, topk=args.topk)

    print("Running role discovery (balanced + inertial sweeps)...")
    role_of_word = cluster_roles_jax(
        vocab_size=V, nbrs_np=nbrs, wts_np=wts,
        n_clusters=int(args.n_clusters), n_sweeps=int(args.n_sweeps), seed=int(args.seed),
        gamma_size=float(GAMMA_SIZE), update_prob=float(UPDATE_PROB), lambda_dir=float(LAMBDA_DIR)
    )

    C = int(args.n_clusters)
    role_stream = role_of_word[ids]  # (T,)
    J_cc = role_directional_coupling(role_of_word, edges_i, edges_j, edges_w, n_clusters=C)

    # Top couplings
    flat = [(float(J_cc[a, b]), a, b) for a in range(C) for b in range(C)]
    flat.sort(key=lambda t: t[0], reverse=True)
    print("\n" + "-" * 80)
    print("TOP ROLE→ROLE DIRECTIONAL COUPLINGS (J_cc)")
    print("-" * 80)
    for v, a, b in flat[:min(20, len(flat))]:
        print(f"c{a:02d} → c{b:02d}   J_cc={v:.4f}")

    # Role labels (global), skipping punctuation tokens
    tok_counts = np.bincount(ids, minlength=V).astype(np.int64)
    role_to_tokens: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(C)}
    for wid in range(1, V):
        role = int(role_of_word[wid])
        role_to_tokens[role].append((int(tok_counts[wid]), wid))
    for r in range(C):
        role_to_tokens[r].sort(reverse=True, key=lambda t: t[0])

    def role_name(r: int) -> str:
        reps = []
        for c, wid in role_to_tokens[r]:
            tok = id_to_tok[wid]
            if tok == "<UNK>":
                continue
            if is_punct(tok):
                continue
            reps.append(tok)
            if len(reps) >= 6:
                break
        return " ".join(reps) if reps else "<empty>"

    print("\n" + "-" * 80)
    print("ROLE LABELS (global top tokens per role; punctuation-skipped)")
    print("-" * 80)
    for r in range(C):
        print(f"c{r:02d}: {role_name(r)}")

    # Sample segments
    rng = np.random.default_rng(args.seed)
    T = len(ids)
    if T < args.seg_len + 2:
        raise ValueError("Input too short for segment analysis; reduce --seg_len or use larger input.")
    starts = rng.integers(0, T - args.seg_len - 1, size=args.n_segs, endpoint=False)

    entities: List[Entity] = []
    next_eid = 0

    # time series
    ts_entity_count = []
    ts_avg_stability = []
    ts_rel_surprise = []
    ts_births = []
    ts_retired = []
    ts_active_roles = []

    # relational matrix tracking by entity id
    prev_rel_by_id: Dict[int, Dict[int, float]] | None = None

    # record some per-segment details for JSON
    seg_details = []

    print("\n" + "=" * 80)
    print("CEBS-STYLE ENTITY REGISTRY ANALYSIS OVER SAMPLED SEGMENTS")
    print("=" * 80)

    for seg_idx, st in enumerate(starts.tolist()):
        seg_roles = role_stream[st: st + args.seg_len].astype(np.int32)
        sig, counts = role_transition_signature(seg_roles, n_clusters=C, laplace=0.25)
        sig = smooth_signatures(sig, J_cc, beta=float(args.msgpass_beta))

        active = np.where(counts >= int(args.min_role_count))[0].astype(np.int32)
        active_set = set(int(x) for x in active.tolist())
        ts_active_roles.append(int(len(active)))
        births = 0
        retired = 0
        stabilities: List[float] = []

        # binding map role->entity id (only active roles get bindings)
        role_to_entity: Dict[int, int] = {}

        if len(active) > 0:
            A = sig[active]  # (R,2C)

            if len(entities) > 0:
                B = np.stack([e.canon for e in entities], axis=0).astype(np.float32)  # (E,2C)
                sims = cosine_rows(A, B)  # (R,E)
                best_e_idx = np.argmax(sims, axis=1)  # (R,)
                best_sim = sims[np.arange(len(active)), best_e_idx]
            else:
                best_e_idx = np.zeros((len(active),), dtype=np.int32)
                best_sim = np.full((len(active),), -1.0, dtype=np.float32)

            # For each active role, retrieve or birth.
            for rr, r in enumerate(active.tolist()):
                sim = float(best_sim[rr])
                if (len(entities) == 0) or (sim < float(args.birth_thresh)):
                    e = Entity(
                        eid=next_eid,
                        canon=A[rr].copy(),
                        last_seen=seg_idx,
                        confidence=0.55,
                        bound_role=int(r),
                        bindings=1,
                    )
                    next_eid += 1
                    entities.append(e)
                    role_to_entity[int(r)] = int(e.eid)
                    births += 1
                    stabilities.append(1.0)  # newborn
                else:
                    e = entities[int(best_e_idx[rr])]
                    e.last_seen = seg_idx
                    e.bound_role = int(r)
                    e.bindings += 1
                    stab = max(0.0, min(1.0, sim))
                    stabilities.append(stab)
                    a = adaptive_alpha(stab)
                    e.canon = (a * e.canon + (1.0 - a) * A[rr]).astype(np.float32)
                    e.confidence = float(min(1.0, 0.985 * e.confidence + 0.015 * stab))
                    role_to_entity[int(r)] = int(e.eid)

        # retire old entities (that haven't been seen in --retire_after segments)
        alive = []
        for e in entities:
            if (seg_idx - e.last_seen) > int(args.retire_after):
                retired += 1
                continue
            alive.append(e)
        entities = alive

        avg_stab = safe_mean(stabilities)

        # Relational satisfaction among entities, summarized as mean absolute delta over common ids
        rel_by_id: Dict[int, Dict[int, float]] = {}
        for ei in entities:
            row = {}
            for ej in entities:
                if ei.bound_role is None or ej.bound_role is None:
                    s = 0.0
                else:
                    j = float(J_cc[int(ei.bound_role), int(ej.bound_role)])
                    s = j * min(1.0, (ei.confidence + ej.confidence) / 2.0)
                row[ej.eid] = float(s)
            rel_by_id[ei.eid] = row

        rel_surprise = 0.0
        if prev_rel_by_id is not None:
            common = sorted(set(prev_rel_by_id.keys()) & set(rel_by_id.keys()))
            if common:
                diffs = []
                for i in common:
                    row_prev = prev_rel_by_id[i]
                    row_now = rel_by_id[i]
                    common2 = set(row_prev.keys()) & set(row_now.keys())
                    for j in common2:
                        diffs.append(abs(row_now[j] - row_prev[j]))
                rel_surprise = float(np.mean(diffs)) if diffs else 0.0
        prev_rel_by_id = rel_by_id

        ts_entity_count.append(int(len(entities)))
        ts_avg_stability.append(float(avg_stab))
        ts_rel_surprise.append(float(rel_surprise))
        ts_births.append(int(births))
        ts_retired.append(int(retired))

        # per-segment role counts and representative tokens (within segment)
        seg_tok_ids = ids[st: st + args.seg_len]
        seg_role_ids = role_of_word[seg_tok_ids]
        seg_role_tok_counts: Dict[int, Dict[int, int]] = {}
        for wid, rr in zip(seg_tok_ids.tolist(), seg_role_ids.tolist()):
            if wid == 0:
                continue
            seg_role_tok_counts.setdefault(int(rr), {})
            seg_role_tok_counts[int(rr)][int(wid)] = seg_role_tok_counts[int(rr)].get(int(wid), 0) + 1

        def seg_role_reps(r: int, k: int = 8) -> str:
            d = seg_role_tok_counts.get(r, {})
            items = sorted(d.items(), key=lambda t: t[1], reverse=True)
            toks = []
            for wid, _c in items:
                tok = id_to_tok[wid]
                if is_punct(tok):
                    continue
                toks.append(tok)
                if len(toks) >= k:
                    break
            return ", ".join(toks)

        # Top role->role transitions by count for this segment
        x = seg_roles[:-1]
        y = seg_roles[1:]
        M = np.zeros((C, C), dtype=np.int32)
        np.add.at(M, (x, y), 1)
        top_pairs = [(int(M[a, b]), a, b) for a in range(C) for b in range(C) if M[a, b] > 0]
        top_pairs.sort(key=lambda t: t[0], reverse=True)

        print("\n" + "-" * 80)
        print(f"[segment {seg_idx+1:02d}/{args.n_segs}]  start={st:,}  len={args.seg_len}  time={now()}")
        print("-" * 80)
        print(f"Active roles (count >= {args.min_role_count}): {len(active)}  |  births: {births}  retired: {retired}")
        if len(active) > 0:
            # show the 8 most frequent active roles
            act_sorted = sorted([(int(counts[r]), int(r)) for r in active.tolist()], reverse=True)[:8]
            act_str = "  ".join([f"c{r:02d}({c})" for c, r in act_sorted])
            print(f"Top active roles by count: {act_str}")
        print(f"Entities alive: {len(entities)}")
        print(f"Avg stability (context overlap): {avg_stab:.4f}")
        trig = "  <<< TRIGGER" if rel_surprise > float(args.surprise_thresh) else ""
        print(f"Relational surprise: {rel_surprise:.4f}{trig}")

        print("\nROLE→ROLE FLOWS (top 10 within this segment)")
        for cnt, a, b in top_pairs[:10]:
            tag = ""
            if (a in active_set) and (b in active_set):
                tag = " (active→active)"
            ea = role_to_entity.get(a, None)
            eb = role_to_entity.get(b, None)
            jv = float(J_cc[a, b])
            ea_s = f"e{ea:03d}" if ea is not None else "--"
            eb_s = f"e{eb:03d}" if eb is not None else "--"
            print(f"  c{a:02d}→c{b:02d}  count={cnt:4d}  J_cc={jv:.3f}  {ea_s}→{eb_s}{tag}")

        # Show example slot fillings: prefer active roles; fall back to random if none
        print("\nROLE 'SLOTS' (example fillings)")
        show_roles = active.tolist()
        rng.shuffle(show_roles)
        show_roles = show_roles[:3]
        if not show_roles:
            show_roles = list(range(C))
            rng.shuffle(show_roles)
            show_roles = show_roles[:3]
        for r in show_roles:
            ent = role_to_entity.get(int(r), None)
            ent_s = f"e{ent:03d}" if ent is not None else "<unbound>"
            reps = seg_role_reps(int(r), k=8)
            print(f"  slot(role)=c{int(r):02d}  entity={ent_s}  seg_reps: {reps if reps else '<none>'}")

        seg_details.append({
            "segment_index": int(seg_idx),
            "start": int(st),
            "length": int(args.seg_len),
            "active_roles": [int(x) for x in active.tolist()],
            "role_counts": {f"c{r:02d}": int(counts[r]) for r in range(C)},
            "births": int(births),
            "retired": int(retired),
            "entities_alive": int(len(entities)),
            "avg_stability": float(avg_stab),
            "relational_surprise": float(rel_surprise),
            "top_role_transitions": [{"from": int(a), "to": int(b), "count": int(cnt)} for cnt, a, b in top_pairs[:25]],
            "role_to_entity": {f"c{r:02d}": int(eid) for r, eid in role_to_entity.items()},
        })

    # Final entity summary
    print("\n" + "=" * 80)
    print("FINAL ENTITY REGISTRY SUMMARY (top 25 by bindings, then confidence)")
    print("=" * 80)
    entities.sort(key=lambda e: (e.bindings, e.confidence), reverse=True)
    for e in entities[:min(25, len(entities))]:
        r = e.bound_role if e.bound_role is not None else -1
        print(f"e{e.eid:03d}  bound_role=c{r:02d}  conf={e.confidence:.3f}  bindings={e.bindings:4d}  last_seen_seg={e.last_seen}")

    # Save JSON + plaintext summary
    report = {
        "config": vars(args),
        "time": now(),
        "vocab_size": int(V),
        "num_tokens_raw": int(len(toks)),
        "in_vocab_tokens": int(in_vocab),
        "num_edges": int(len(edges_i)),
        "n_clusters": int(C),
        "role_labels_top_tokens": {f"c{r:02d}": role_name(r) for r in range(C)},
        "J_cc_top20": [{"from": int(a), "to": int(b), "J": float(v)} for v, a, b in flat[:20]],
        "registry_timeseries": {
            "entity_count": ts_entity_count,
            "avg_stability": ts_avg_stability,
            "relational_surprise": ts_rel_surprise,
            "births": ts_births,
            "retired": ts_retired,
            "active_roles": ts_active_roles,
        },
        "segments": seg_details,
        "entities_final": [
            {
                "eid": int(e.eid),
                "bound_role": (int(e.bound_role) if e.bound_role is not None else None),
                "confidence": float(e.confidence),
                "bindings": int(e.bindings),
                "last_seen_seg": int(e.last_seen),
                "canon": e.canon.astype(np.float32).tolist(),
            }
            for e in entities
        ],
    }
    json_path = os.path.join(args.outdir, "step6a_results.json")
    save_json(json_path, report)
    print(f"\nWrote JSON report: {json_path}")

    # Plaintext summary: quickly greppable
    txt_path = os.path.join(args.outdir, "step6a_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("HNET STEP 6A SUMMARY\n")
        f.write(f"time: {report['time']}\n")
        f.write(f"vocab_size: {V}\n")
        f.write(f"edges: {len(edges_i)}\n")
        f.write("\nROLE LABELS:\n")
        for r in range(C):
            f.write(f"c{r:02d}: {role_name(r)}\n")
        f.write("\nTOP J_cc:\n")
        for row in report["J_cc_top20"]:
            f.write(f"c{row['from']:02d} -> c{row['to']:02d}   J={row['J']:.4f}\n")
        f.write("\nFINAL ENTITIES (top 50 by bindings/conf):\n")
        for e in entities[:min(50, len(entities))]:
            r = e.bound_role if e.bound_role is not None else -1
            f.write(f"e{e.eid:03d}  role=c{r:02d}  conf={e.confidence:.3f}  bindings={e.bindings}  last_seen={e.last_seen}\n")
    print(f"Wrote text summary: {txt_path}")

    # Plots
    xs = np.arange(len(ts_entity_count), dtype=np.int32)

    def save_plot(y, title, fname, ylabel):
        plt.figure()
        plt.plot(xs, y)
        plt.title(title)
        plt.xlabel("sampled segment index")
        plt.ylabel(ylabel)
        plt.tight_layout()
        path = os.path.join(args.outdir, fname)
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"Wrote plot: {path}")

    save_plot(ts_entity_count, "Entity count over segments", "entity_count.png", "entities")
    save_plot(ts_avg_stability, "Avg stability (context overlap) over segments", "avg_stability.png", "cosine overlap")
    save_plot(ts_rel_surprise, "Relational surprise over segments", "relational_surprise.png", "mean |Δ satisfaction|")
    save_plot(ts_births, "Entity births over segments", "births.png", "births")
    save_plot(ts_retired, "Entity retirements over segments", "retired.png", "retired")
    save_plot(ts_active_roles, "Active roles per segment", "active_roles.png", "active roles")

    print("\nDone.")


if __name__ == "__main__":
    main()
