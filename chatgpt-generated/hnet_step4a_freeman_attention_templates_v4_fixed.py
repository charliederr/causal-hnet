import math
import random
from collections import Counter, defaultdict

import numpy as np
import jax
import jax.numpy as jnp

print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

# ============================================================
# Configuration
# ============================================================

# Text / vocab
MIN_FREQ = 5
MAX_VOCAB = 20000
COOC_WINDOW = 6                    # directional word->word window for J

# Step 2 (directional Hamiltonian clustering)
N_CLUSTERS = 16
N_SWEEPS = 40
GAMMA_SIZE = 1.0                   # size penalty strength (prevents giant cluster)
UPDATE_PROB = 0.20                 # inertia: update only this fraction per sweep

# Embedding / attention dims
EMB_DIM = 64
HEAD_DIM = 64

# Freeman-template attention diagnostics
SEG_LEN = 256
N_SEGS = 8
ATTN_LOCAL_WINDOW = 64             # restrict attention to +/- this window
TOKEN_NOISE = 0.01                 # small noise (keeps ties from dominating)

# Bias weights
ALPHA_TEMPLATE = 1.0               # template compatibility strength
BETA_DIRECTIONAL = 0.75            # directional causal bias strength

# Reporting
TOP_TEMPLATE_PAIRS = 16
TOP_TOKEN_LINKS = 20
TOP_JCC = 20
RANDOM_SEED = 0

random.seed(RANDOM_SEED)

# ============================================================
# Helpers
# ============================================================

def softmax_rowwise(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

# ============================================================
# Load text
# ============================================================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
T = len(tokens)

print("=" * 80)
print("HNET STEP 4A — FREEMAN ATTENTION TEMPLATES (V4: FIXED)")
print("=" * 80)
print(f"Total tokens: {T}")

# ============================================================
# Vocabulary
# ============================================================

freq = Counter(tokens)
vocab = [w for w, c in freq.items() if c >= MIN_FREQ]
vocab = vocab[:MAX_VOCAB]
word_to_id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

print(f"Vocabulary size: {V}")

# Token ids (-1 = OOV)
tok_ids = np.full((T,), -1, dtype=np.int32)
for i, w in enumerate(tokens):
    tok_ids[i] = word_to_id.get(w, -1)

# ============================================================
# Build directional word->word co-occurrence edges (CPU)
# ============================================================

cooc = defaultdict(float)
for i in range(T):
    wi = tok_ids[i]
    if wi < 0:
        continue
    end = min(i + COOC_WINDOW, T)
    for j in range(i + 1, end):
        wj = tok_ids[j]
        if wj < 0:
            continue
        cooc[(wi, wj)] += 1.0

norm_out = np.zeros(V, dtype=np.float64)
norm_in = np.zeros(V, dtype=np.float64)
for (i, j), v in cooc.items():
    norm_out[i] += v
    norm_in[j] += v

edges_i, edges_j, edges_w = [], [], []
for (i, j), v in cooc.items():
    w = v / math.sqrt(norm_out[i] * norm_in[j] + 1e-8)
    edges_i.append(i)
    edges_j.append(j)
    edges_w.append(w)

edges_i = jnp.array(edges_i, dtype=jnp.int32)
edges_j = jnp.array(edges_j, dtype=jnp.int32)
edges_w = jnp.array(edges_w, dtype=jnp.float32)

E = int(edges_i.shape[0])
print(f"Directional edges (word->word): {E}")

# ============================================================
# Step 2: Balanced + inertial directional clustering (JAX)
# ============================================================

@jax.jit
def balanced_inertial_sweep(state, key):
    """
    score_{i,c} = sum_{i->j} w_ij * [s_j=c]  - GAMMA_SIZE * log(count_c)
    then inertial update: only UPDATE_PROB fraction updated
    """
    S = jax.nn.one_hot(state, N_CLUSTERS)                 # (V,C)

    # neighbor influence field (directional)
    contrib = edges_w[:, None] * S[edges_j]               # (E,C)
    field = jnp.zeros((V, N_CLUSTERS)).at[edges_i].add(contrib)

    # size penalty
    counts = jnp.sum(S, axis=0) + 1.0                     # (C,)
    penalty = GAMMA_SIZE * jnp.log(counts)[None, :]       # (1,C)

    score = field - penalty
    best = jnp.argmax(score, axis=1)                      # (V,)

    # inertial mask: update only a fraction of nodes each sweep
    mask = jax.random.bernoulli(key, p=UPDATE_PROB, shape=best.shape)
    new_state = jnp.where(mask, best, state)
    return new_state

key = jax.random.PRNGKey(RANDOM_SEED)
state = jax.random.randint(key, (V,), 0, N_CLUSTERS)

print("Running BALANCED + INERTIAL directional sweeps...")
for s in range(N_SWEEPS):
    key, sub = jax.random.split(key)
    state = balanced_inertial_sweep(state, sub)
    if (s + 1) % 5 == 0 or s == 0:
        counts = np.array(jnp.bincount(state, length=N_CLUSTERS))
        nonzero = int((counts > 0).sum())
        print(f"  sweep {s+1}/{N_SWEEPS}  nonempty_clusters={nonzero}  counts={counts.tolist()}")

state_np = np.array(state)

# ============================================================
# Build cluster-to-cluster directional couplings J_cc (CPU then JAX)
# ============================================================

J_cc_np = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float64)
for (wi, wj), v in cooc.items():
    ci = state_np[wi]
    cj = state_np[wj]
    J_cc_np[ci, cj] += v

# Normalize to [0,1] by max
if J_cc_np.max() > 0:
    J_cc_np = J_cc_np / (J_cc_np.max() + 1e-12)

J_cc = jnp.array(J_cc_np, dtype=jnp.float32)

print("\n" + "-" * 80)
print("TOP CLUSTER→CLUSTER DIRECTIONAL COUPLINGS (J_cc)")
print("-" * 80)
pairs = []
for a in range(N_CLUSTERS):
    for b in range(N_CLUSTERS):
        pairs.append((J_cc_np[a, b], a, b))
pairs.sort(reverse=True)
for val, a, b in pairs[:TOP_JCC]:
    print(f"c{a:02d} → c{b:02d}   J_cc={val:.4f}")

# ============================================================
# Embeddings: word + cluster (avoid vanishing means)
# ============================================================

key, sub1, sub2, sub3 = jax.random.split(key, 4)

# word embeddings
Xw = jax.random.normal(sub1, (V, EMB_DIM)) * 0.2

# cluster embeddings (template prototypes)
Xc = jax.random.normal(sub2, (N_CLUSTERS, EMB_DIM)) * 0.5

# template compatibility W_psi from cluster embedding cosine similarity
Xc_norm = Xc / (jnp.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
W_psi = Xc_norm @ Xc_norm.T      # (C,C)

# attention projections
Wq = jax.random.normal(sub3, (EMB_DIM, HEAD_DIM)) / math.sqrt(EMB_DIM)
key, sub4 = jax.random.split(key)
Wk = jax.random.normal(sub4, (EMB_DIM, HEAD_DIM)) / math.sqrt(EMB_DIM)

# ============================================================
# Segment attention (Freeman-style): hard templates + biases
# ============================================================

@jax.jit
def segment_attention(word_ids_seg, noise_key):
    """
    word_ids_seg: (S,) word ids (int32)
    returns:
      A: (S,S) attention weights
      base_logits: (S,S)
      tpl_bias: (S,S) template compat bias
      dir_bias: (S,S) directional cluster bias
      c_hard: (S,) cluster ids
      pair_mass: (C,C) = onehot(c)^T A onehot(c)
    """
    c_hard = state[word_ids_seg]  # (S,)

    # token embedding: word + its cluster template embedding (+ tiny noise)
    eps = TOKEN_NOISE * jax.random.normal(noise_key, (word_ids_seg.shape[0], EMB_DIM))
    Eseg = Xw[word_ids_seg] + Xc[c_hard] + eps

    Q = Eseg @ Wq
    K = Eseg @ Wk
    base_logits = (Q @ K.T) / jnp.sqrt(HEAD_DIM)

    # local attention mask
    S = word_ids_seg.shape[0]
    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    mask = dist <= ATTN_LOCAL_WINDOW  # (S,S)

    # hard template bias lookup
    tpl_bias = W_psi[c_hard[:, None], c_hard[None, :]]
    dir_bias = J_cc[c_hard[:, None], c_hard[None, :]]

    logits = base_logits + ALPHA_TEMPLATE * tpl_bias + BETA_DIRECTIONAL * dir_bias

    # mask out self + far positions
    logits = jnp.where(mask, logits, -jnp.inf)
    logits = logits - 1e9 * jnp.eye(S, dtype=logits.dtype)

    A = jax.nn.softmax(logits, axis=1)

    # template flow: onehot(c)^T A onehot(c)
    onehot = jax.nn.one_hot(c_hard, N_CLUSTERS)  # (S,C)
    pair_mass = onehot.T @ A @ onehot            # (C,C)

    return A, base_logits, tpl_bias, dir_bias, c_hard, pair_mass

# ============================================================
# Sample segments and run diagnostics
# ============================================================

if T < SEG_LEN + 1:
    raise RuntimeError("Text too short for SEG_LEN.")

starts = [random.randint(0, T - SEG_LEN - 1) for _ in range(N_SEGS)]

print("\nCompiling segment attention (one-time JAX cost)...")
dummy = tok_ids[starts[0]:starts[0] + SEG_LEN].copy()
dummy[dummy < 0] = 0
_ = segment_attention(jnp.array(dummy, dtype=jnp.int32), key)[0].block_until_ready()
print("Compilation finished.\n")

template_flow = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float64)
token_link_records = []

for si, start in enumerate(starts, 1):
    seg = tok_ids[start:start + SEG_LEN].copy()
    seg[seg < 0] = 0
    seg_j = jnp.array(seg, dtype=jnp.int32)

    key, nk = jax.random.split(key)
    A, baseL, tplB, dirB, c_hard, pair_mass = segment_attention(seg_j, nk)

    template_flow += np.array(pair_mass)

    A_np = np.array(A)
    base_np = np.array(baseL)
    tpl_np = np.array(tplB)
    dir_np = np.array(dirB)
    c_np = np.array(c_hard)

    # top token-to-token links by attention weight
    flat = A_np.ravel()
    top_idx = np.argpartition(flat, -TOP_TOKEN_LINKS)[-TOP_TOKEN_LINKS:]
    top_idx = top_idx[np.argsort(-flat[top_idx])]

    for idx in top_idx:
        i = int(idx // SEG_LEN)
        j = int(idx % SEG_LEN)
        if i == j:
            continue
        wi = int(seg[i])
        wj = int(seg[j])
        dist = abs((start + i) - (start + j))
        token_link_records.append({
            "seg": si,
            "pos_i": start + i,
            "pos_j": start + j,
            "w_i": vocab[wi],
            "w_j": vocab[wj],
            "attn": float(A_np[i, j]),
            "base": float(base_np[i, j]),
            "tpl": float(tpl_np[i, j]),
            "dir": float(dir_np[i, j]),
            "c_i": int(c_np[i]),
            "c_j": int(c_np[j]),
            "dist": dist
        })

print("-" * 80)
print("TEMPLATE→TEMPLATE ATTENTION FLOWS (aggregated over sampled segments)")
print("-" * 80)

template_flow /= max(1, N_SEGS)
pairs = []
for a in range(N_CLUSTERS):
    for b in range(N_CLUSTERS):
        if a == b:
            continue
        pairs.append((template_flow[a, b], a, b))
pairs.sort(reverse=True)

for val, a, b in pairs[:TOP_TEMPLATE_PAIRS]:
    print(f"Template {a:02d} → Template {b:02d}   flow={val:.4f}   J_cc={J_cc_np[a,b]:.4f}   Wpsi={float(W_psi[a,b]):.4f}")

print("\n" + "-" * 80)
print("TOP TOKEN→TOKEN LINKS (with contribution breakdown)")
print("-" * 80)

token_link_records.sort(key=lambda r: r["attn"], reverse=True)

for r in token_link_records[:TOP_TOKEN_LINKS]:
    near = "LOCAL" if r["dist"] <= ATTN_LOCAL_WINDOW else "LONG"
    print(
        f"[seg {r['seg']}] {near:>4}  "
        f"{r['w_i']:<14} → {r['w_j']:<14}  "
        f"attn={r['attn']:.4f}  "
        f"base={r['base']:+.3f} tpl={ALPHA_TEMPLATE*r['tpl']:+.3f} dir={BETA_DIRECTIONAL*r['dir']:+.3f}  "
        f"(c{r['c_i']:02d}→c{r['c_j']:02d}, dist={r['dist']})"
    )

print("\nDone.")
