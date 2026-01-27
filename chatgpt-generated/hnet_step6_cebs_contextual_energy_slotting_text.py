#!/usr/bin/env python3
"""
HNET STEP 6 — CEBS-INSPIRED CONTEXTUAL ENERGY-BASED SLOTTING (TEXT EDITION)

What this adds (inspired by the CEBS paper):
  - Explicit "entity registry" over time (birth / persistence / retirement)
  - Context signatures as primary evidence for identity (here: role-transition context vectors)
  - Partial bindings (roles can be unbound) + stability diagnostics
  - Relational "Hamiltonian" style energy and "relational surprise" triggers

This file is designed to be:
  - single-file, runnable locally
  - fast (JAX-jitted clustering; vectorized signature computation)
  - verbose (prints extensive CLI diagnostics)
  - produces JSON output + a few plots

Usage:
  python hnet_step6_cebs_contextual_energy_slotting_text.py --input input.txt
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
# Config defaults (override via CLI)
# =============================================================================

DEFAULT_N_CLUSTERS = 16
DEFAULT_N_SWEEPS = 40
DEFAULT_TOPK = 64            # top outgoing edges per token for sparse assoc
DEFAULT_SEG_LEN = 256
DEFAULT_N_SEGS = 32          # sampled segments for CEBS registry analysis
DEFAULT_SEED = 0

# clustering dynamics
LAMBDA_DIR = 0.8             # weight of directional association
GAMMA_SIZE = 1.0             # size-balancing penalty
UPDATE_PROB = 0.25           # stochastic update rate per sweep

# registry dynamics
BIRTH_THRESH = 0.72          # cosine overlap threshold for retrieving existing entity
RETIRE_AFTER = 10            # retire entities not seen in this many segments
ALPHA_BASE = 0.90            # EMA baseline for canonical context update
STABILITY_THRESH = 0.65      # stability threshold used for adaptive alpha
ALPHA_BETA = 8.0             # sharpness for alpha adaptation

# relational surprise
SURPRISE_THRESH = 0.18       # triggers a "relational change" event (heuristic)


# =============================================================================
# Utilities
# =============================================================================

def now() -> str:
    import datetime as _dt
    return _dt.datetime.now().isoformat(sep=" ", timespec="seconds")

def tokenize(text: str) -> List[str]:
    # Keep punctuation-ish tokens but split reasonably.
    # This is deliberately simple and fast.
    return re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", text)

def read_text(path: str, max_chars: int | None = None) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if max_chars is not None:
        txt = txt[:max_chars]
    return txt

def build_vocab(tokens: List[str], min_count: int = 2) -> Tuple[Dict[str, int], List[str], np.ndarray]:
    from collections import Counter
    cnt = Counter(tokens)
    # Reserve 0 for <UNK>
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

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))

def softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


# =============================================================================
# Fast directional co-occurrence (token i -> token j, weighted by distance)
# =============================================================================

def build_directional_edges(ids: np.ndarray, max_dist: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build weighted directed edges i->j for pairs within max_dist.
    Weight decays with distance.

    Returns:
      edges_i, edges_j, edges_w (all np arrays, int32/int32/float32)
    """
    t0 = time.time()
    n = len(ids)
    # We'll count within a sliding window using a dictionary of (i,j)->w.
    # For speed, do it chunked and with numpy operations per offset.
    from collections import defaultdict
    acc = defaultdict(float)
    for d in range(1, max_dist + 1):
        w = 1.0 / math.sqrt(d)
        src = ids[:-d]
        dst = ids[d:]
        # aggregate counts per (src,dst)
        # Use numpy unique on packed pairs.
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
    print(f"Building directional co-occurrence (fast)...")
    print(f"Cooc unique edges: {E:,}  (time {dt:.2f}s)")
    return edges_i, edges_j, edges_w


# =============================================================================
# Sparse assoc (top-k outgoing edges per token) to speed clustering objective
# =============================================================================

def build_sparse_assoc(vocab_size: int, edges_i: np.ndarray, edges_j: np.ndarray, edges_w: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each token i, keep top-k outgoing neighbors j with highest weight.

    Returns:
      nbrs: (V, topk) int32
      wts : (V, topk) float32
    """
    t0 = time.time()
    # bucket edges by src
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
# Step-2 style balanced + inertial clustering (roles)
#
# IMPORTANT JAX FIX:
#   - n_clusters must be STATIC inside jit to avoid ConcretizationTypeError with one_hot.
#   - We close over N_CLUSTERS (a Python int), and do NOT pass it as a traced argument.
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
    N_CLUSTERS = n_clusters

    nbrs = jnp.array(nbrs_np, dtype=jnp.int32)  # (V,topk)
    wts = jnp.array(wts_np, dtype=jnp.float32)  # (V,topk)

    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    # init: random
    state0 = jax.random.randint(k0, shape=(vocab_size,), minval=0, maxval=N_CLUSTERS, dtype=jnp.int32)

    def one_hot_state(state):
        # (V,C) using eye indexing; C is static here.
        return jnp.eye(N_CLUSTERS, dtype=jnp.float32)[state]

    def energy_terms(state):
        """
        Compute per-token energy contributions for each candidate cluster.
        E_token[c] = -lambda_dir * sum_k w(i,k) * 1[state[nbr]==c] + size_penalty[c]
        """
        S = one_hot_state(state)  # (V,C)
        # neighbor cluster indicators: gather S at nbrs -> (V,topk,C)
        Sn = S[nbrs]  # (V,topk,C)
        # weighted sum over neighbors -> (V,C)
        assoc = jnp.einsum("vk,vkc->vc", wts, Sn)
        # size balancing: encourage equal counts
        counts = jnp.sum(S, axis=0)  # (C,)
        target = vocab_size / float(N_CLUSTERS)
        size_pen = gamma_size * (counts - target) / (target + 1e-6)  # (C,)
        # broadcast size penalty to each token
        E = -lambda_dir * assoc + size_pen[None, :]
        return E, counts

    @jax.jit
    def sweep(state, key):
        E, counts = energy_terms(state)  # (V,C), (C,)
        # stochastic updates: sample mask
        key, kmask, kchoice = jax.random.split(key, 3)
        mask = jax.random.bernoulli(kmask, p=update_prob, shape=(vocab_size,))
        # choose argmin energy
        new_state = jnp.argmin(E, axis=1).astype(jnp.int32)
        state = jnp.where(mask, new_state, state)
        return state, key, counts

    counts_hist = []
    state = state0
    for s in range(n_sweeps):
        state, key, counts = sweep(state, key)
        if (s == 0) or ((s + 1) % 10 == 0) or (s == n_sweeps - 1):
            counts_np = np.array(counts, dtype=np.float32)
            nonempty = int(np.sum(counts_np > 0.5))
            print(f"  sweep {s+1}/{n_sweeps}  nonempty={nonempty}")
            counts_hist.append(counts_np.tolist())

    return np.array(state, dtype=np.int32)


# =============================================================================
# Context signatures (CEBS-like): role transition context vectors per segment
# =============================================================================

def role_transition_signature(role_ids: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    role_ids: (S,) ints in [0,C)
    Returns:
      sig: (C, 2C) where for each role r:
        [outgoing_probs_to_roles (C), incoming_probs_from_roles (C)]
    """
    C = n_clusters
    x = role_ids[:-1]
    y = role_ids[1:]
    # transition counts: (C,C)
    M = np.zeros((C, C), dtype=np.float32)
    np.add.at(M, (x, y), 1.0)
    # outgoing/incoming normalized
    out = M / (M.sum(axis=1, keepdims=True) + 1e-6)
    inn = M / (M.sum(axis=0, keepdims=True) + 1e-6)
    sig = np.concatenate([out, inn.T], axis=1)  # (C, 2C)
    return sig

def role_directional_coupling(role_of_word: np.ndarray, edges_i: np.ndarray, edges_j: np.ndarray, edges_w: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Aggregate token-level directed edges into role->role coupling matrix J_cc.
    """
    C = n_clusters
    ri = role_of_word[edges_i]
    rj = role_of_word[edges_j]
    J = np.zeros((C, C), dtype=np.float64)
    np.add.at(J, (ri, rj), edges_w.astype(np.float64))
    # normalize to [0,1] by max
    mx = float(J.max()) if J.size else 1.0
    if mx > 0:
        J = J / mx
    return J.astype(np.float32)


# =============================================================================
# Entity Registry (CEBS-like)
# =============================================================================

@dataclass
class Entity:
    eid: int
    canon: np.ndarray          # (2C,) canonical signature
    last_seen: int
    confidence: float
    bound_role: int | None     # which role this entity currently represents (for interpretability)

def adaptive_alpha(stability: float) -> float:
    # if stability is low, update faster (alpha smaller)
    # alpha = ALPHA_BASE + (1-ALPHA_BASE) * sigmoid(beta*(stability - thresh))
    sig = 1.0 / (1.0 + math.exp(-ALPHA_BETA * (stability - STABILITY_THRESH)))
    return float(ALPHA_BASE + (1.0 - ALPHA_BASE) * sig)

def tnorm_product(a: float, b: float) -> float:
    return a * b

def relational_satisfaction(J_cc: np.ndarray, e_src: Entity, e_tgt: Entity) -> float:
    """
    Toy fuzzy satisfaction: high if role->role coupling is high AND both entities are confident.
    """
    if e_src.bound_role is None or e_tgt.bound_role is None:
        return 0.0
    j = float(J_cc[e_src.bound_role, e_tgt.bound_role])
    return float(tnorm_product(j, min(1.0, (e_src.confidence + e_tgt.confidence) / 2.0)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--outdir", type=str, default="step6_out")
    ap.add_argument("--max_chars", type=int, default=0, help="If >0, truncate input text to this many chars")
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--max_dist", type=int, default=128)

    ap.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    ap.add_argument("--n_sweeps", type=int, default=DEFAULT_N_SWEEPS)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--seg_len", type=int, default=DEFAULT_SEG_LEN)
    ap.add_argument("--n_segs", type=int, default=DEFAULT_N_SEGS)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 80)
    print("HNET STEP 6 — CEBS-INSPIRED CONTEXTUAL ENERGY-BASED SLOTTING (TEXT)")
    print("=" * 80)

    txt = read_text(args.input, max_chars=(args.max_chars if args.max_chars > 0 else None))
    tokens = tokenize(txt)
    print(f"Loaded: {args.input}")
    print(f"Raw tokens: {len(tokens):,}")

    vocab, id_to_tok, ids = build_vocab(tokens, min_count=args.min_count)
    V = len(id_to_tok)
    print(f"Vocab size (min_count={args.min_count}): {V:,}")
    print(f"In-vocab token stream length: {int(np.sum(ids != 0)):,}")

    # Build edges
    edges_i, edges_j, edges_w = build_directional_edges(ids, max_dist=args.max_dist)
    print(f"Edges arrays: E={len(edges_i):,}")

    # Sparse assoc for clustering
    nbrs, wts = build_sparse_assoc(V, edges_i, edges_j, edges_w, topk=args.topk)

    # Cluster roles
    print("Running role discovery (balanced + inertial sweeps)...")
    role_of_word = cluster_roles_jax(
        vocab_size=V,
        nbrs_np=nbrs,
        wts_np=wts,
        n_clusters=int(args.n_clusters),
        n_sweeps=int(args.n_sweeps),
        seed=int(args.seed),
        gamma_size=float(GAMMA_SIZE),
        update_prob=float(UPDATE_PROB),
        lambda_dir=float(LAMBDA_DIR),
    )

    # Couplings at role level (used for "relational energy/surprise")
    J_cc = role_directional_coupling(role_of_word, edges_i, edges_j, edges_w, n_clusters=args.n_clusters)

    # Print top role couplings
    print("\n" + "-" * 80)
    print("TOP ROLE→ROLE DIRECTIONAL COUPLINGS (J_cc)")
    print("-" * 80)
    flat = []
    C = args.n_clusters
    for a in range(C):
        for b in range(C):
            flat.append((float(J_cc[a, b]), a, b))
    flat.sort(reverse=True, key=lambda t: t[0])
    for v, a, b in flat[:min(20, len(flat))]:
        print(f"c{a:02d} → c{b:02d}   J_cc={v:.4f}")

    # Sample segments for CEBS-style registry analysis
    rng = np.random.default_rng(args.seed)
    total = len(ids)
    if total < args.seg_len + 2:
        raise ValueError("Input too short for segment analysis; reduce --seg_len or use larger input.")
    starts = rng.integers(0, total - args.seg_len - 1, size=args.n_segs, endpoint=False)

    # Entities represent "persistent context-defined role bindings"
    entities: List[Entity] = []
    next_eid = 0

    # Time series diagnostics
    t_entity_count = []
    t_avg_stability = []
    t_rel_surprise = []
    t_births = []
    t_retired = []

    # For interpretability: map role -> representative tokens
    role_to_tokens: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(C)}
    # count per token id
    tok_counts = np.bincount(ids, minlength=V).astype(np.int64)
    for wid in range(1, V):
        role = int(role_of_word[wid])
        role_to_tokens[role].append((int(tok_counts[wid]), wid))
    for r in range(C):
        role_to_tokens[r].sort(reverse=True, key=lambda t: t[0])

    def role_name(r: int) -> str:
        reps = role_to_tokens[r][:5]
        parts = []
        for c, wid in reps:
            parts.append(id_to_tok[wid])
        return " ".join(parts) if parts else "<empty>"

    print("\n" + "-" * 80)
    print("ROLE LABELS (top freq tokens per role)")
    print("-" * 80)
    for r in range(C):
        print(f"c{r:02d}: {role_name(r)}")

    # Precompute role stream for all ids (including <UNK>=0, we keep it but it gets a role too)
    role_stream = role_of_word[ids]  # (T,)

    print("\n" + "=" * 80)
    print("CEBS-STYLE ENTITY REGISTRY ANALYSIS OVER SAMPLED SEGMENTS")
    print("=" * 80)

    prev_rel_matrix = None

    for t_idx, st in enumerate(starts.tolist()):
        seg_roles = role_stream[st: st + args.seg_len].astype(np.int32)
        seg_sig = role_transition_signature(seg_roles, n_clusters=C)  # (C,2C)

        births = 0
        retired = 0
        stabilities = []

        # For each role r, decide which entity it corresponds to in this segment.
        # Binding is partial/optional: if no good match, it stays unbound and may spawn a new entity.
        role_to_entity: Dict[int, int] = {}

        for r in range(C):
            sig_r = seg_sig[r].astype(np.float32)
            # find best matching entity among those whose bound_role == r or None (flexible)
            best_e = None
            best_sim = -1.0
            for e in entities:
                sim = cosine(sig_r, e.canon)
                if sim > best_sim:
                    best_sim = sim
                    best_e = e

            if best_e is None or best_sim < BIRTH_THRESH:
                # birth new entity for this role
                e = Entity(eid=next_eid, canon=sig_r.copy(), last_seen=t_idx, confidence=0.5, bound_role=r)
                next_eid += 1
                entities.append(e)
                role_to_entity[r] = e.eid
                births += 1
                stabilities.append(1.0)  # newborn stability
            else:
                # bind and update canonical signature
                best_e.last_seen = t_idx
                best_e.bound_role = r
                # stability based on overlap
                stab = best_sim
                stabilities.append(stab)
                a = adaptive_alpha(stab)
                best_e.canon = a * best_e.canon + (1.0 - a) * sig_r
                best_e.confidence = float(min(1.0, 0.98 * best_e.confidence + 0.02 * stab))
                role_to_entity[r] = best_e.eid

        # retire old entities
        alive = []
        for e in entities:
            if (t_idx - e.last_seen) > RETIRE_AFTER:
                retired += 1
                continue
            alive.append(e)
        entities = alive

        avg_stab = float(np.mean(stabilities)) if stabilities else 0.0

        # Relational "surprise": compare satisfaction matrix across segments
        rel = np.zeros((len(entities), len(entities)), dtype=np.float32)
        for i, ei in enumerate(entities):
            for j, ej in enumerate(entities):
                rel[i, j] = relational_satisfaction(J_cc, ei, ej)

        rel_surprise = 0.0
        if prev_rel_matrix is not None:
            # align by entity id where possible; simplest: compare overlap on min size
            m = min(prev_rel_matrix.shape[0], rel.shape[0])
            if m > 0:
                rel_surprise = float(np.mean(np.abs(rel[:m, :m] - prev_rel_matrix[:m, :m])))
        prev_rel_matrix = rel

        t_entity_count.append(len(entities))
        t_avg_stability.append(avg_stab)
        t_rel_surprise.append(rel_surprise)
        t_births.append(births)
        t_retired.append(retired)

        # Print per-segment summary
        print("\n" + "-" * 80)
        print(f"[segment {t_idx+1:02d}/{args.n_segs}]  start={st:,}  len={args.seg_len}  time={now()}")
        print("-" * 80)
        print(f"Entities alive: {len(entities):d}   births: {births:d}   retired: {retired:d}")
        print(f"Avg stability (context overlap): {avg_stab:.4f}")
        print(f"Relational surprise: {rel_surprise:.4f}" + ("  <<< TRIGGER" if rel_surprise > SURPRISE_THRESH else ""))

        # Top "slot/role fillings": show role->entity and top transitions inside segment
        # Compute top transitions by count
        x = seg_roles[:-1]
        y = seg_roles[1:]
        M = np.zeros((C, C), dtype=np.int32)
        np.add.at(M, (x, y), 1)
        top_pairs = []
        for a in range(C):
            for b in range(C):
                if M[a, b] > 0:
                    top_pairs.append((int(M[a, b]), a, b))
        top_pairs.sort(reverse=True, key=lambda t: t[0])

        print("\nROLE→ROLE FLOWS (top 10 within this segment)")
        for cnt, a, b in top_pairs[:10]:
            ea = role_to_entity.get(a, -1)
            eb = role_to_entity.get(b, -1)
            jv = float(J_cc[a, b])
            print(f"  c{a:02d}→c{b:02d}  count={cnt:4d}  J_cc={jv:.3f}  e{ea:03d}→e{eb:03d}")

        # Show a few interpretability snippets: pick 3 roles and show representative tokens
        pick_roles = list(range(C))
        rng.shuffle(pick_roles)
        pick_roles = pick_roles[:3]
        print("\nROLE 'SLOTS' (example fillings)")
        for r in pick_roles:
            reps = role_to_tokens[r][:8]
            rep_toks = ", ".join([f"{id_to_tok[wid]}" for _, wid in reps])
            print(f"  slot(role)=c{r:02d}  entity=e{role_to_entity[r]:03d}  reps: {rep_toks}")

    # Final entity summary
    print("\n" + "=" * 80)
    print("FINAL ENTITY REGISTRY SUMMARY")
    print("=" * 80)
    entities.sort(key=lambda e: e.confidence, reverse=True)
    for e in entities[:min(20, len(entities))]:
        r = e.bound_role if e.bound_role is not None else -1
        print(f"e{e.eid:03d}  bound_role=c{r:02d}  conf={e.confidence:.3f}  last_seen_seg={e.last_seen}")

    # Save JSON report
    report = {
        "config": vars(args),
        "time": now(),
        "vocab_size": V,
        "num_tokens_raw": len(tokens),
        "num_edges": int(len(edges_i)),
        "n_clusters": int(C),
        "role_labels_top_tokens": {f"c{r:02d}": role_name(r) for r in range(C)},
        "J_cc_top20": [{"from": int(a), "to": int(b), "J": float(v)} for v, a, b in flat[:20]],
        "registry_timeseries": {
            "entity_count": t_entity_count,
            "avg_stability": t_avg_stability,
            "relational_surprise": t_rel_surprise,
            "births": t_births,
            "retired": t_retired,
        },
        "entities_final": [
            {
                "eid": int(e.eid),
                "bound_role": (int(e.bound_role) if e.bound_role is not None else None),
                "confidence": float(e.confidence),
                "last_seen_seg": int(e.last_seen),
                "canon": e.canon.astype(np.float32).tolist(),
            }
            for e in entities
        ],
    }
    json_path = os.path.join(args.outdir, "step6_results.json")
    save_json(json_path, report)
    print(f"\nWrote JSON report: {json_path}")

    # Plots
    xs = np.arange(len(t_entity_count))

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

    save_plot(t_entity_count, "Entity count over segments", "entity_count.png", "entities")
    save_plot(t_avg_stability, "Avg stability (context overlap) over segments", "avg_stability.png", "cosine overlap")
    save_plot(t_rel_surprise, "Relational surprise over segments", "relational_surprise.png", "mean |Δ satisfaction|")
    save_plot(t_births, "Entity births over segments", "births.png", "births")
    save_plot(t_retired, "Entity retirements over segments", "retired.png", "retired")

    print("\nDone.")


if __name__ == "__main__":
    main()
