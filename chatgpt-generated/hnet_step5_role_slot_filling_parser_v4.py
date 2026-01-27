#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import random
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, DefaultDict, Optional

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================
# Config
# ============================================================

SEED = 0
INPUT_PATH = "input.txt"

MIN_FREQ = 2
MAX_VOCAB = 20000

COOC_WINDOW = 6

N_CLUSTERS = 16
N_SWEEPS = 40
GAMMA_SIZE = 1.0
UPDATE_PROB = 0.20

ASSOC_TOPK = 64

ALPHA_COMPOSE = 0.5
BETA_COMPOSE = 0.5
LAMBDA_ROLE = 0.50
LOCALITY_PRIOR = 0.05

SEGMENT_LEN = 64
N_SEGMENTS_FOR_SLOTS = 80
N_PARSE_EXAMPLES = 3
MAX_PARSE_LEN = 80

TOP_SLOTS = 20
TOP_FILLERS_PER_SLOT = 20
TOP_CONTEXTS_PER_SLOT = 10
TOP_SPAN_FILLERS_PER_SLOT = 15

OUT_JSON = "slot_report.json"


# ============================================================
# Utilities
# ============================================================

def log(msg: str):
    print(msg, flush=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def take_segments(rng: np.random.RandomState, stream: np.ndarray, seg_len: int, n: int) -> List[np.ndarray]:
    if len(stream) <= seg_len + 2:
        return [stream[:seg_len]]
    max_start = len(stream) - seg_len - 1
    starts = rng.randint(0, max_start, size=(n,))
    return [stream[s:s+seg_len] for s in starts]

def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, va in a.items():
        na += va * va
        vb = b.get(k, 0.0)
        dot += va * vb
    for vb in b.values():
        nb += vb * vb
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / math.sqrt(na * nb)


# ============================================================
# Tokenize + vocab
# ============================================================

def load_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower().split()

def build_vocab(tokens: List[str], min_freq: int, max_vocab: int):
    freq = Counter(tokens)
    vocab = [w for w, c in freq.items() if c >= min_freq]
    vocab = vocab[:max_vocab]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i, freq

def to_stream(tokens: List[str], w2i: Dict[str, int]) -> np.ndarray:
    return np.array([w2i[w] for w in tokens if w in w2i], dtype=np.int32)


# ============================================================
# Fast directional co-occurrence via packed keys + np.unique
# ============================================================

def build_cooc_fast(stream: np.ndarray, V: int, window: int):
    t0 = time.time()
    packed_counts: DefaultDict[int, int] = defaultdict(int)

    if len(stream) < 2:
        return {}, np.zeros(V, np.float64), np.zeros(V, np.float64), time.time() - t0

    for d in range(1, window):
        a = stream[:-d].astype(np.int64)
        b = stream[d:].astype(np.int64)
        keys = a * np.int64(V) + b
        uniq, cnt = np.unique(keys, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            packed_counts[k] += int(c)

    cooc: Dict[Tuple[int, int], float] = {}
    out_mass = np.zeros(V, dtype=np.float64)
    in_mass = np.zeros(V, dtype=np.float64)

    for k, c in packed_counts.items():
        i = int(k // V)
        j = int(k - i * V)
        cooc[(i, j)] = float(c)
        out_mass[i] += float(c)
        in_mass[j] += float(c)

    return cooc, out_mass, in_mass, time.time() - t0


def cooc_to_edges(cooc: Dict[Tuple[int, int], float], out_mass: np.ndarray, in_mass: np.ndarray):
    edges_i, edges_j, edges_w = [], [], []
    for (i, j), v in cooc.items():
        w = v / math.sqrt(out_mass[i] * in_mass[j] + 1e-8)
        edges_i.append(i)
        edges_j.append(j)
        edges_w.append(w)
    return (
        jnp.array(edges_i, dtype=jnp.int32),
        jnp.array(edges_j, dtype=jnp.int32),
        jnp.array(edges_w, dtype=jnp.float32),
    )


def build_sparse_assoc_out(cooc: Dict[Tuple[int, int], float], out_mass: np.ndarray, V: int, topk: int):
    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(V)]
    for (i, j), v in cooc.items():
        w = v / math.sqrt(out_mass[i] + 1e-8)
        buckets[i].append((j, float(w)))

    assoc = []
    for i in range(V):
        lst = buckets[i]
        lst.sort(key=lambda x: x[1], reverse=True)
        assoc.append(lst[:topk])
    return assoc


# ============================================================
# Balanced inertial clustering (JAX) — v4 FIX
#   - No n_clusters argument passes through jit
#   - N_CLUSTERS is a module constant used inside
# ============================================================

@jax.jit
def balanced_inertial_sweep(state, key, edges_i, edges_j, edges_w):
    # N_CLUSTERS is compile-time constant here
    S = jax.nn.one_hot(state, N_CLUSTERS)               # (V,C)
    contrib = edges_w[:, None] * S[edges_j]             # (E,C)
    field = jnp.zeros((S.shape[0], N_CLUSTERS)).at[edges_i].add(contrib)

    counts = jnp.sum(S, axis=0) + 1.0
    penalty = GAMMA_SIZE * jnp.log(counts)[None, :]
    score = field - penalty
    best = jnp.argmax(score, axis=1)

    mask = jax.random.bernoulli(key, p=UPDATE_PROB, shape=best.shape)
    return jnp.where(mask, best, state)

def cluster_roles(V: int, edges_i, edges_j, edges_w, n_sweeps: int):
    key = jax.random.PRNGKey(SEED)
    state = jax.random.randint(key, (V,), 0, N_CLUSTERS).astype(jnp.int32)

    log("Running role discovery (balanced + inertial sweeps)...")
    t0 = time.time()
    for s in range(n_sweeps):
        key, sub = jax.random.split(key)
        state = balanced_inertial_sweep(state, sub, edges_i, edges_j, edges_w)
        if (s + 1) % 10 == 0 or s == 0 or (s + 1) == n_sweeps:
            counts = np.array(jnp.bincount(state, length=N_CLUSTERS))
            nonzero = int((counts > 0).sum())
            log(f"  sweep {s+1:>2}/{n_sweeps}  nonempty={nonzero}  counts(min/med/max)={counts.min()}/{int(np.median(counts))}/{counts.max()}")
    log(f"Role discovery time: {time.time() - t0:.2f}s")
    return np.array(state)

def cluster_to_cluster_coupling(cooc: Dict[Tuple[int, int], float], role_of_word: np.ndarray):
    J = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float64)
    for (i, j), v in cooc.items():
        ci = int(role_of_word[i])
        cj = int(role_of_word[j])
        J[ci, cj] += v
    if J.max() > 0:
        J = J / (J.max() + 1e-12)
    return J


# ============================================================
# Parsing
# ============================================================

@dataclass
class SpanAssoc:
    out: Dict[int, float]
    role_hist: np.ndarray

@dataclass
class Node:
    l: int
    r: int
    role: int
    score: float
    left: Optional["Node"] = None
    right: Optional["Node"] = None

def assoc_word(word_id: int, role_id: int, assoc_out_sparse: List[List[Tuple[int, float]]]) -> SpanAssoc:
    out = {j: float(w) for (j, w) in assoc_out_sparse[word_id]}
    hist = np.zeros((N_CLUSTERS,), dtype=np.float64)
    hist[role_id] += 1.0
    return SpanAssoc(out=out, role_hist=hist)

def compose_assoc(a: SpanAssoc, b: SpanAssoc, alpha: float, beta: float) -> SpanAssoc:
    out: Dict[int, float] = {}
    for k, v in a.out.items():
        out[k] = out.get(k, 0.0) + alpha * v
    for k, v in b.out.items():
        out[k] = out.get(k, 0.0) + beta * v
    hist = alpha * a.role_hist + beta * b.role_hist
    return SpanAssoc(out=out, role_hist=hist)

def span_role_id(role_hist: np.ndarray) -> int:
    return int(np.argmax(role_hist))

def parse_segment(seg: np.ndarray, role_of_word: np.ndarray, assoc_out_sparse: List[List[Tuple[int, float]]], J: np.ndarray, vocab: List[str]) -> Node:
    n = len(seg)
    best_node = [[None for _ in range(n+1)] for _ in range(n)]
    best_assoc = [[None for _ in range(n+1)] for _ in range(n)]

    for i in range(n):
        w = int(seg[i])
        c = int(role_of_word[w])
        a = assoc_word(w, c, assoc_out_sparse)
        best_assoc[i][i+1] = a
        best_node[i][i+1] = Node(l=i, r=i+1, role=c, score=0.0)

    for length in range(2, n+1):
        for l in range(0, n - length + 1):
            r = l + length
            best_s = -1e9
            best_k = None
            best_a = None
            best_left = None
            best_right = None
            for k in range(l+1, r):
                left_a = best_assoc[l][k]
                right_a = best_assoc[k][r]
                if left_a is None or right_a is None:
                    continue
                sim = cosine_sparse(left_a.out, right_a.out)
                rl = span_role_id(left_a.role_hist)
                rr = span_role_id(right_a.role_hist)
                compat = float(J[rl, rr])
                prior = -LOCALITY_PRIOR * float(length)
                s = sim + LAMBDA_ROLE * compat + prior
                if s > best_s:
                    best_s = s
                    best_k = k
                    best_a = compose_assoc(left_a, right_a, ALPHA_COMPOSE, BETA_COMPOSE)
                    best_left = best_node[l][k]
                    best_right = best_node[k][r]
            if best_k is None:
                best_k = l+1
                left_a = best_assoc[l][best_k]
                right_a = best_assoc[best_k][r]
                best_a = compose_assoc(left_a, right_a, ALPHA_COMPOSE, BETA_COMPOSE)
                best_s = -1e6
                best_left = best_node[l][best_k]
                best_right = best_node[best_k][r]
            role_here = span_role_id(best_a.role_hist)
            best_assoc[l][r] = best_a
            best_node[l][r] = Node(l=l, r=r, role=role_here, score=float(best_s), left=best_left, right=best_right)

    return best_node[0][n]

def render_tree(node: Node, seg: np.ndarray, vocab: List[str], indent: int = 0) -> str:
    pad = "  " * indent
    if node.left is None or node.right is None:
        w = vocab[int(seg[node.l])]
        return f"{pad}[{node.l}:{node.r}] c{node.role:02d}  {w}\n"
    span_text = " ".join(vocab[int(t)] for t in seg[node.l:node.r])
    if len(span_text) > 100:
        span_text = span_text[:97] + "..."
    s = f'{pad}[{node.l}:{node.r}] c{node.role:02d} score={node.score:+.3f}  "{span_text}"\n'
    s += render_tree(node.left, seg, vocab, indent+1)
    s += render_tree(node.right, seg, vocab, indent+1)
    return s


# ============================================================
# Slot diagnostics
# ============================================================

def span_fillers_for_slot(segments: List[np.ndarray], role_of_word: np.ndarray, vocab: List[str], slot: Tuple[int, int], max_span: int = 6):
    ci, cj = slot
    spans = Counter()
    for seg in segments:
        rseq = [int(role_of_word[int(w)]) for w in seg]
        words = [vocab[int(w)] for w in seg]
        for i in range(len(seg)):
            if rseq[i] != ci:
                continue
            for j in range(i+1, min(len(seg), i+1+max_span)):
                if rseq[j] == cj:
                    spans[" ".join(words[i:j+1])] += 1
    return spans

def slot_stats_from_segments(segments: List[np.ndarray], role_of_word: np.ndarray, vocab: List[str], J: np.ndarray):
    slot_counts = defaultdict(float)
    slot_src_words = defaultdict(Counter)
    slot_tgt_words = defaultdict(Counter)
    slot_contexts = defaultdict(Counter)

    for seg in segments:
        rseq = [int(role_of_word[int(w)]) for w in seg]
        for t in range(len(seg)-1):
            wi = int(seg[t])
            wj = int(seg[t+1])
            ci = rseq[t]
            cj = rseq[t+1]
            slot_counts[(ci, cj)] += 1.0
            slot_src_words[(ci, cj)][wi] += 1
            slot_tgt_words[(ci, cj)][wj] += 1
            left_role = rseq[t-1] if t-1 >= 0 else -1
            right_role = rseq[t+2] if t+2 < len(seg) else -1
            sig = f"{left_role}->{ci}->{cj}->{right_role}"
            slot_contexts[(ci, cj)][sig] += 1

    scored = []
    for (ci, cj), cnt in slot_counts.items():
        score = float(cnt) * (0.25 + float(J[ci, cj]))
        scored.append((score, ci, cj, float(cnt), float(J[ci, cj])))
    scored.sort(reverse=True)
    return scored, slot_src_words, slot_tgt_words, slot_contexts

def print_slot_report(scored_slots, slot_src_words, slot_tgt_words, slot_contexts, segments, role_of_word, vocab, J):
    log("\n" + "=" * 80)
    log("SLOT / ROLE-FILLING REPORT (unsupervised)")
    log("=" * 80)

    report = {"top_slots": []}

    for rank, (score, ci, cj, cnt, jcc) in enumerate(scored_slots[:TOP_SLOTS], start=1):
        log("\n" + "-" * 80)
        log(f"[{rank:02d}] SLOT: c{ci:02d} → c{cj:02d}   score={score:.2f}   count={cnt:.0f}   J_cc={jcc:.3f}")
        log("-" * 80)

        src_top = slot_src_words[(ci, cj)].most_common(TOP_FILLERS_PER_SLOT)
        tgt_top = slot_tgt_words[(ci, cj)].most_common(TOP_FILLERS_PER_SLOT)
        ctx_top = slot_contexts[(ci, cj)].most_common(TOP_CONTEXTS_PER_SLOT)

        log("Top SOURCE fillers:")
        for w, c in src_top:
            log(f"  {vocab[w]:<16}  n={c}")
        log("\nTop TARGET fillers:")
        for w, c in tgt_top:
            log(f"  {vocab[w]:<16}  n={c}")

        log("\nMost diagnostic local ROLE contexts (left->ci->cj->right):")
        for sig, c in ctx_top:
            log(f"  {sig:<18}  n={c}")

        span_counts = span_fillers_for_slot(segments, role_of_word, vocab, (ci, cj), max_span=6)
        span_top = span_counts.most_common(TOP_SPAN_FILLERS_PER_SLOT)
        if span_top:
            log("\nTop SPAN fillers bridging c_i ... c_j:")
            for s, c in span_top:
                log(f'  "{s}"   n={c}')

        report["top_slots"].append({
            "rank": rank,
            "slot": [int(ci), int(cj)],
            "score": float(score),
            "count": float(cnt),
            "J_cc": float(jcc),
            "top_src_words": [{"w": vocab[w], "n": int(c)} for w, c in src_top],
            "top_tgt_words": [{"w": vocab[w], "n": int(c)} for w, c in tgt_top],
            "top_contexts": [{"sig": sig, "n": int(c)} for sig, c in ctx_top],
            "top_span_fillers": [{"span": s, "n": int(c)} for s, c in span_top],
        })

    return report


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    log("=" * 80)
    log("HNET STEP 5 (v4) — START")
    log("=" * 80)

    if not os.path.exists(INPUT_PATH):
        log(f"ERROR: {INPUT_PATH} not found in cwd={os.getcwd()}")
        out_path = os.path.abspath(OUT_JSON)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"error": "input_missing", "cwd": os.getcwd()}, f, indent=2)
        log(f"Wrote: {out_path}")
        return

    size = os.path.getsize(INPUT_PATH)
    log(f"Input: {INPUT_PATH}  size={size/1024/1024:.2f} MB  cwd={os.getcwd()}")

    t0 = time.time()
    tokens = load_tokens(INPUT_PATH)
    log(f"Loaded tokens: {len(tokens):,}  (time {time.time()-t0:.2f}s)")

    vocab, w2i, freq = build_vocab(tokens, MIN_FREQ, MAX_VOCAB)
    log(f"Vocab size: {len(vocab):,}   MIN_FREQ={MIN_FREQ}")

    stream = to_stream(tokens, w2i)
    log(f"In-vocab stream length: {len(stream):,}")

    report = {"meta": {"cwd": os.getcwd(), "input_path": INPUT_PATH, "input_size_bytes": size,
                       "raw_tokens": len(tokens), "vocab_size": len(vocab), "stream_len": int(len(stream)),
                       "min_freq": MIN_FREQ, "n_clusters": N_CLUSTERS}}

    if len(vocab) == 0 or len(stream) < 100:
        log("WARNING: stream too small after vocab filtering; not enough data to compute slots/roles.")
        out_path = os.path.abspath(OUT_JSON)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(f"Wrote: {out_path}")
        return

    log("\nBuilding directional co-occurrence (fast)...")
    cooc, out_mass, in_mass, dt = build_cooc_fast(stream, len(vocab), COOC_WINDOW)
    log(f"Cooc unique edges: {len(cooc):,}  (time {dt:.2f}s)")

    edges_i, edges_j, edges_w = cooc_to_edges(cooc, out_mass, in_mass)
    log(f"Edges arrays: E={int(edges_i.shape[0]):,}")

    assoc_out_sparse = build_sparse_assoc_out(cooc, out_mass, len(vocab), ASSOC_TOPK)
    log(f"Sparse assoc built: topk={ASSOC_TOPK}")

    role_of_word = cluster_roles(len(vocab), edges_i, edges_j, edges_w, N_SWEEPS)

    log("\n" + "=" * 80)
    log("ROLE SUMMARIES (top words by corpus frequency)")
    log("=" * 80)
    by_role = [[] for _ in range(N_CLUSTERS)]
    vocab_freq = {w: freq[w] for w in vocab}
    for wid in range(len(vocab)):
        by_role[int(role_of_word[wid])].append(wid)
    for c in range(N_CLUSTERS):
        words = by_role[c]
        words_sorted = sorted(words, key=lambda wid: vocab_freq[vocab[wid]], reverse=True)
        top = words_sorted[:15]
        top_str = ", ".join(vocab[wid] for wid in top)
        log(f"c{c:02d}  size={len(words):4d}  top={top_str}")

    J = cluster_to_cluster_coupling(cooc, role_of_word)

    log("\n" + "=" * 80)
    log("TOP ROLE→ROLE COMPATIBILITIES (J_cc)")
    log("=" * 80)
    pairs = []
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            pairs.append((float(J[i, j]), i, j))
    pairs.sort(reverse=True)
    report["J_cc_top"] = []
    for val, i, j in pairs[:30]:
        log(f"c{i:02d} → c{j:02d}  J_cc={val:.4f}")
        report["J_cc_top"].append({"i": int(i), "j": int(j), "J": float(val)})

    rng = np.random.RandomState(SEED + 77)
    segments = take_segments(rng, stream, SEGMENT_LEN, N_SEGMENTS_FOR_SLOTS)

    scored_slots, slot_src_words, slot_tgt_words, slot_contexts = slot_stats_from_segments(
        segments, role_of_word, vocab, J
    )
    report.update(print_slot_report(
        scored_slots, slot_src_words, slot_tgt_words, slot_contexts,
        segments, role_of_word, vocab, J
    ))

    log("\n" + "=" * 80)
    log("PARSING EXAMPLES (binary tree by association similarity + role-compat)")
    log("=" * 80)
    report["parse_examples"] = []
    for ex in range(min(N_PARSE_EXAMPLES, len(segments))):
        seg = segments[ex][:min(len(segments[ex]), MAX_PARSE_LEN)]
        preview = " ".join(vocab[int(w)] for w in seg[:40])
        if len(preview) > 140:
            preview = preview[:137] + "..."
        log("\n" + "-" * 80)
        log(f"Example {ex+1}: preview: {preview}")
        node = parse_segment(seg, role_of_word, assoc_out_sparse, J, vocab)
        tree_str = render_tree(node, seg, vocab)
        log(tree_str)
        report["parse_examples"].append({"preview": preview, "tree": tree_str})

    out_path = os.path.abspath(OUT_JSON)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"\nWrote: {out_path}")
    log("Done.")


if __name__ == "__main__":
    main()
