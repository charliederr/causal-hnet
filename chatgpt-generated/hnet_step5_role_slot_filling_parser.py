#!/usr/bin/env python3
"""
HNET STEP 5 — ROLE / SLOT-FILLING DIAGNOSTICS + TOP-DOWN PARSING (UNSUPERVISED)

Implements the core ideas from "Hierarchical Parsing with Association Templates":
- explicit association vectors (from directional co-occurrence)
- templates (roles) as latent groupings over association vectors
- parsing by recursive split selection using association similarity + role compatibility

Outputs:
- extensive console report (slots, fillers, contexts, parse tree examples)
- slot_report.json (machine-readable summary)

Notes:
- Templates are used as *constraints/compatibility*, not labels for parse nodes.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, DefaultDict
from collections import Counter, defaultdict

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================
# Config (tune here)
# ============================================================

SEED = 0

INPUT_PATH = "input.txt"

MIN_FREQ = 5
MAX_VOCAB = 20000

# directional co-occurrence window
COOC_WINDOW = 6

# roles/templates
N_CLUSTERS = 16
N_SWEEPS = 40
GAMMA_SIZE = 1.0
UPDATE_PROB = 0.20

# association vector construction for spans:
# we represent each token/span by a sparse vector over vocab (outgoing assoc)
# For spans we compose: A(span)=alpha*A(left)+beta*A(right)
ALPHA_COMPOSE = 0.5
BETA_COMPOSE = 0.5

# parse scoring
LAMBDA_ROLE = 0.50      # strength of role compatibility term
LOCALITY_PRIOR = 0.05   # penalize long-distance merges slightly
MAX_PARSE_LEN = 80      # parse only first N tokens of chosen segments for display
N_PARSE_EXAMPLES = 3
SEGMENT_LEN = 64
N_SEGMENTS_FOR_SLOTS = 80   # sample segments for slot/filler stats

# diagnostics sizes
TOP_SLOTS = 20
TOP_FILLERS_PER_SLOT = 20
TOP_CONTEXTS_PER_SLOT = 10
TOP_SPAN_FILLERS_PER_SLOT = 15

OUT_JSON = "slot_report.json"


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def softmax(x: np.ndarray):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def safe_log(x: float) -> float:
    return math.log(x + 1e-12)

def take_segments(rng: np.random.RandomState, tokens: np.ndarray, seg_len: int, n: int) -> List[np.ndarray]:
    if len(tokens) <= seg_len + 2:
        return [tokens[:seg_len]]
    max_start = len(tokens) - seg_len - 1
    starts = rng.randint(0, max_start, size=(n,))
    return [tokens[s:s+seg_len] for s in starts]


# ============================================================
# Data + vocab
# ============================================================

def load_and_tokenize(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    return text.split()

def build_vocab(tokens: List[str], min_freq: int, max_vocab: int):
    freq = Counter(tokens)
    vocab = [w for w, c in freq.items() if c >= min_freq]
    vocab = vocab[:max_vocab]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i

def to_stream(tokens: List[str], w2i: Dict[str, int]) -> np.ndarray:
    return np.array([w2i[w] for w in tokens if w in w2i], dtype=np.int32)


# ============================================================
# Association structure from directional co-occurrence
# ============================================================

def build_directional_edges(stream: np.ndarray, vocab_size: int, window: int):
    """
    Returns:
      edges_i, edges_j, edges_w: arrays for edges i->j with normalized weights.
      cooc_dict: dict[(i,j)] = raw count
      out_norm, in_norm: normalization masses
    """
    cooc: DefaultDict[Tuple[int, int], float] = defaultdict(float)
    for t in range(len(stream)):
        wi = int(stream[t])
        end = min(t + window, len(stream))
        for u in range(t + 1, end):
            wj = int(stream[u])
            cooc[(wi, wj)] += 1.0

    out_mass = np.zeros(vocab_size, dtype=np.float64)
    in_mass = np.zeros(vocab_size, dtype=np.float64)
    for (i, j), v in cooc.items():
        out_mass[i] += v
        in_mass[j] += v

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
        cooc,
        out_mass,
        in_mass,
    )

def build_sparse_assoc_out(cooc: Dict[Tuple[int, int], float], out_mass: np.ndarray, vocab_size: int, topk: int = 64):
    """
    For each word i, build a sparse outgoing association vector A(i): list of (j, weight).
    Weight is normalized directional association i->j.
    Keeps topk targets per i for speed.
    """
    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(vocab_size)]
    for (i, j), v in cooc.items():
        w = v / math.sqrt(out_mass[i] * (1.0) + 1e-8)  # slightly different norm; ok for diagnostics
        buckets[i].append((j, w))

    assoc = []
    for i in range(vocab_size):
        lst = buckets[i]
        lst.sort(key=lambda x: x[1], reverse=True)
        assoc.append(lst[:topk])
    return assoc


# ============================================================
# Balanced inertial clustering (roles/templates) in JAX
# ============================================================

@jax.jit
def balanced_inertial_sweep(state, key, edges_i, edges_j, edges_w, n_clusters: int, gamma_size: float, update_prob: float):
    S = jax.nn.one_hot(state, n_clusters)               # (V,C)
    contrib = edges_w[:, None] * S[edges_j]             # (E,C)
    field = jnp.zeros((S.shape[0], n_clusters)).at[edges_i].add(contrib)

    counts = jnp.sum(S, axis=0) + 1.0
    penalty = gamma_size * jnp.log(counts)[None, :]
    score = field - penalty
    best = jnp.argmax(score, axis=1)

    mask = jax.random.bernoulli(key, p=update_prob, shape=best.shape)
    return jnp.where(mask, best, state)

def cluster_roles(vocab_size: int, edges_i, edges_j, edges_w, n_clusters: int, n_sweeps: int):
    key = jax.random.PRNGKey(SEED)
    state = jax.random.randint(key, (vocab_size,), 0, n_clusters)

    print("Running role discovery (balanced + inertial sweeps)...")
    for s in range(n_sweeps):
        key, sub = jax.random.split(key)
        state = balanced_inertial_sweep(state, sub, edges_i, edges_j, edges_w, n_clusters, GAMMA_SIZE, UPDATE_PROB)
        if (s + 1) % 10 == 0 or s == 0 or (s + 1) == n_sweeps:
            counts = np.array(jnp.bincount(state, length=n_clusters))
            nonzero = int((counts > 0).sum())
            print(f"  sweep {s+1}/{n_sweeps}  nonempty={nonzero}  counts(min/med/max)={counts.min()}/{int(np.median(counts))}/{counts.max()}")
    return state

def cluster_to_cluster_coupling(cooc: Dict[Tuple[int, int], float], role_of_word: np.ndarray, n_clusters: int):
    J = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for (i, j), v in cooc.items():
        ci = int(role_of_word[i])
        cj = int(role_of_word[j])
        J[ci, cj] += v
    if J.max() > 0:
        J = J / (J.max() + 1e-12)
    return J


# ============================================================
# Span association vectors + similarity
# ============================================================

@dataclass
class SpanAssoc:
    # sparse outgoing assoc: dict[target_word] = weight
    out: Dict[int, float]
    # role distribution: counts over roles among words in span
    role_hist: np.ndarray

def assoc_word(word_id: int, role_id: int, assoc_out_sparse: List[List[Tuple[int, float]]], n_clusters: int) -> SpanAssoc:
    out = {j: float(w) for (j, w) in assoc_out_sparse[word_id]}
    hist = np.zeros((n_clusters,), dtype=np.float64)
    hist[role_id] += 1.0
    return SpanAssoc(out=out, role_hist=hist)

def compose_assoc(a: SpanAssoc, b: SpanAssoc, alpha: float, beta: float) -> SpanAssoc:
    out: Dict[int, float] = {}
    # merge sparse dicts
    for k, v in a.out.items():
        out[k] = out.get(k, 0.0) + alpha * v
    for k, v in b.out.items():
        out[k] = out.get(k, 0.0) + beta * v

    hist = alpha * a.role_hist + beta * b.role_hist
    return SpanAssoc(out=out, role_hist=hist)

def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate smaller
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

def span_role_id(role_hist: np.ndarray) -> int:
    return int(np.argmax(role_hist))

def role_compat(role_left: int, role_right: int, J: np.ndarray) -> float:
    return float(J[role_left, role_right])


# ============================================================
# Top-down parsing (actually CKY-style best split; prints top-down tree)
# ============================================================

@dataclass
class Node:
    l: int
    r: int
    role: int
    score: float
    left: "Node | None" = None
    right: "Node | None" = None

def parse_segment(tokens: np.ndarray,
                  roles: np.ndarray,
                  assoc_out_sparse: List[List[Tuple[int, float]]],
                  J: np.ndarray,
                  vocab: List[str]) -> Node:
    """
    CKY-like dynamic programming to choose best binary tree by score:
      S(l,k,r) = cos(A(l,k),A(k,r)) + lambda*J(role_l,role_r) - locality_prior*(r-l)
    """
    n = len(tokens)
    # DP tables
    best_node = [[None for _ in range(n+1)] for _ in range(n)]
    best_assoc = [[None for _ in range(n+1)] for _ in range(n)]

    # base spans
    for i in range(n):
        w = int(tokens[i])
        c = int(roles[w])
        a = assoc_word(w, c, assoc_out_sparse, N_CLUSTERS)
        best_assoc[i][i+1] = a
        best_node[i][i+1] = Node(l=i, r=i+1, role=c, score=0.0)

    # lengths
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
                compat = role_compat(rl, rr, J)
                prior = -LOCALITY_PRIOR * float(length)
                s = sim + LAMBDA_ROLE * compat + prior
                if s > best_s:
                    best_s = s
                    best_k = k
                    best_a = compose_assoc(left_a, right_a, ALPHA_COMPOSE, BETA_COMPOSE)
                    best_left = best_node[l][k]
                    best_right = best_node[k][r]
            if best_k is None:
                # fallback: just merge adjacent
                best_k = l+1
                left_a = best_assoc[l][best_k]
                right_a = best_assoc[best_k][r]
                best_a = compose_assoc(left_a, right_a, ALPHA_COMPOSE, BETA_COMPOSE)
                best_s = -1e6
                best_left = best_node[l][best_k]
                best_right = best_node[best_k][r]

            role_here = span_role_id(best_a.role_hist)
            best_assoc[l][r] = best_a
            best_node[l][r] = Node(l=l, r=r, role=role_here, score=best_s, left=best_left, right=best_right)

    return best_node[0][n]

def render_tree(node: Node, tokens: np.ndarray, vocab: List[str], indent: int = 0) -> str:
    pad = "  " * indent
    if node.left is None or node.right is None:
        w = vocab[int(tokens[node.l])]
        return f"{pad}[{node.l}:{node.r}] c{node.role:02d}  {w}\n"
    else:
        span_text = " ".join(vocab[int(t)] for t in tokens[node.l:node.r])
        if len(span_text) > 80:
            span_text = span_text
