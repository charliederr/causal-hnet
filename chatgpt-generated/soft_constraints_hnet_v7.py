import math
import random
import re
from collections import Counter, defaultdict
import numpy as np

import jax
import jax.numpy as jnp

# ============================================================
# Configuration
# ============================================================

MIN_FREQ = 5
MAX_VOCAB = 20000
COOC_WINDOW = 6

N_CLUSTERS = 16
N_SWEEPS = 40

TAU_START = 1.25
TAU_END = 0.12
DAMPING = 0.35
GAMMA_SIZE = 1.25

# Embeddings
EMB_DIM = 96              # a bit larger helps n-grams
ETA_UNARY = 0.75

# Char n-gram hashing
NGRAM_MIN = 3
NGRAM_MAX = 5
HASH_BUCKETS = 2**18      # 262k buckets; CPU memory OK
NGRAM_SCALE = 0.2         # overall magnitude

# Attention diagnostics
HEAD_DIM = 64
SEG_LEN = 256
N_SEGS = 8
ATTN_LOCAL_WINDOW = 64
TOKEN_NOISE = 0.01

ALPHA_TEMPLATE = 1.0
BETA_DIRECTIONAL = 0.75
LAMBDA_IDF_TARGET = 0.15

TOP_JCC = 20
TOP_TEMPLATE_PAIRS = 16
TOP_TOKEN_LINKS = 20

# Output filtering
MIN_IDF_TO_PRINT = 1.8
SKIP_SELF_LINKS = True
SKIP_PUNCT_HEAVY = True

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Utils
# ============================================================

_punct_re = re.compile(r"^[\W_]+$", re.UNICODE)

def is_punct_heavy(token: str) -> bool:
    # True if token is entirely non-alphanumeric (punctuation) or very short punctuation-like
    if _punct_re.match(token):
        return True
    # also treat tokens like "so." "hump," as punct-heavy if most chars are punct
    alnum = sum(ch.isalnum() for ch in token)
    return alnum <= 1 and len(token) <= 4

def normalize_for_ngrams(token: str) -> str:
    # lowercase, strip surrounding punctuation
    t = token.lower()
    # keep internal apostrophes but strip other punctuation at ends
    t = re.sub(r"^[^a-z0-9]+", "", t)
    t = re.sub(r"[^a-z0-9]+$", "", t)
    return t

def char_ngrams(s: str, nmin: int, nmax: int):
    if not s:
        return []
    s2 = f"<{s}>"
    out = []
    L = len(s2)
    for n in range(nmin, nmax + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            out.append(s2[i:i+n])
    return out

def stable_hash32(x: str) -> int:
    # deterministic hash independent of Python's randomized hash seed
    h = 2166136261
    for ch in x:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

# ============================================================
# Header / devices
# ============================================================

print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

# ============================================================
# Load text
# ============================================================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
T = len(tokens)

print("=" * 80)
print("HNET STEP 4A — FREEMAN ATTENTION TEMPLATES (SOFT TEMPLATES v7: NGRAM EMBEDDINGS)")
print("=" * 80)
print(f"Total tokens: {T}")

# ============================================================
# Vocabulary + token ids
# ============================================================

freq = Counter(tokens)
vocab = [w for w, c in freq.items() if c >= MIN_FREQ]
vocab = vocab[:MAX_VOCAB]
word_to_id = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"Vocabulary size: {V}")

tok_ids = np.full((T,), -1, dtype=np.int32)
for i, w in enumerate(tokens):
    tok_ids[i] = word_to_id.get(w, -1)

# IDF over token stream
counts_vocab = np.zeros(V, dtype=np.int64)
for tid in tok_ids:
    if tid >= 0:
        counts_vocab[tid] += 1
N = counts_vocab.sum()
idf = np.log((N + 1.0) / (counts_vocab + 1.0)) + 1.0
idf = idf.astype(np.float32)
idf_j = jnp.array(idf)

# ============================================================
# Directional co-occurrence edges (CPU one-time)
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
# Build char n-gram hashed word embeddings (CPU -> JAX)
# ============================================================

rng = np.random.RandomState(RANDOM_SEED)
bucket_vecs = rng.normal(size=(HASH_BUCKETS, EMB_DIM)).astype(np.float32)
bucket_vecs *= (NGRAM_SCALE / math.sqrt(EMB_DIM))

Xw_np = np.zeros((V, EMB_DIM), dtype=np.float32)

for wid, token in enumerate(vocab):
    normed = normalize_for_ngrams(token)
    ngs = char_ngrams(normed, NGRAM_MIN, NGRAM_MAX)
    if not ngs:
        # fallback: random but small
        Xw_np[wid] = rng.normal(size=(EMB_DIM,)).astype(np.float32) * 0.01
        continue
    idxs = [stable_hash32(g) % HASH_BUCKETS for g in ngs]
    Xw_np[wid] = bucket_vecs[idxs].mean(axis=0)

Xw = jnp.array(Xw_np)

# Cluster prototypes for unary symmetry breaking
key = jax.random.PRNGKey(RANDOM_SEED)
key, subz = jax.random.split(key)
Zc = jax.random.normal(subz, (N_CLUSTERS, EMB_DIM)) * 0.5

UNARY = ETA_UNARY * (Xw @ Zc.T)

Zc_norm = Zc / (jnp.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
W_psi = Zc_norm @ Zc_norm.T

# ============================================================
# Soft clustering
# ============================================================

def tau_at(sweep: int) -> float:
    t = sweep / max(1, (N_SWEEPS - 1))
    return TAU_START * ((TAU_END / TAU_START) ** t)

@jax.jit
def mean_entropy(P):
    return -jnp.mean(jnp.sum(P * jnp.log(P + 1e-12), axis=1))

@jax.jit
def soft_sweep(P, tau):
    contrib = edges_w[:, None] * P[edges_j]
    field = jnp.zeros((V, N_CLUSTERS), dtype=P.dtype).at[edges_i].add(contrib)

    counts = jnp.sum(P, axis=0) + 1.0
    penalty = GAMMA_SIZE * jnp.log(counts)[None, :]

    score = field + UNARY - penalty
    P_new = jax.nn.softmax(score / tau, axis=1)

    P_mixed = (1.0 - DAMPING) * P + DAMPING * P_new
    P_mixed = P_mixed / (jnp.sum(P_mixed, axis=1, keepdims=True) + 1e-12)
    return P_mixed

P = jnp.ones((V, N_CLUSTERS), dtype=jnp.float32) / float(N_CLUSTERS)

print("Running SOFT (mean-field + anneal + ngram embeddings) directional sweeps...")
for s in range(N_SWEEPS):
    tau = tau_at(s)
    P = soft_sweep(P, tau)
    if (s + 1) % 5 == 0 or s == 0:
        counts_soft = np.array(jnp.sum(P, axis=0))
        ent = float(mean_entropy(P))
        hard = np.array(jnp.argmax(P, axis=1))
        counts_hard = np.bincount(hard, minlength=N_CLUSTERS)
        print(f"  sweep {s+1}/{N_SWEEPS}  tau={tau:.3f}  mean_entropy={ent:.3f}  "
              f"soft_counts(min/med/max)=({counts_soft.min():.1f}/{np.median(counts_soft):.1f}/{counts_soft.max():.1f})  "
              f"argmax_nonempty={int((counts_hard>0).sum())}")

# ============================================================
# J_cc and PMI contrast
# ============================================================

Pi = P[edges_i]
Pj = P[edges_j]
Pi_w = Pi * edges_w[:, None]

J_mass = (Pi_w.T @ Pj)
J_row = J_mass / (jnp.sum(J_mass, axis=1, keepdims=True) + 1e-12)

pi = jnp.sum(P, axis=0) / float(V)
base = pi[:, None] * pi[None, :]
J_pmi = jnp.log((J_row + 1e-12) / (base + 1e-12))
J_pmi = J_pmi - jnp.mean(J_pmi)
J_pmi = J_pmi / (jnp.std(J_pmi) + 1e-6)

J_pmi_np = np.array(J_pmi)

print("\n" + "-" * 80)
print("TOP CLUSTER→CLUSTER DIRECTIONAL COUPLINGS (J_pmi)  [SOFT contrastive]")
print("-" * 80)
pairs = []
for a in range(N_CLUSTERS):
    for b in range(N_CLUSTERS):
        pairs.append((J_pmi_np[a, b], a, b))
pairs.sort(reverse=True)
for val, a, b in pairs[:TOP_JCC]:
    print(f"c{a:02d} → c{b:02d}   J_pmi={val:.3f}")

# ============================================================
# Attention
# ============================================================

key, subq, subk = jax.random.split(key, 3)
Wq = jax.random.normal(subq, (EMB_DIM, HEAD_DIM)) / math.sqrt(EMB_DIM)
Wk = jax.random.normal(subk, (EMB_DIM, HEAD_DIM)) / math.sqrt(EMB_DIM)

@jax.jit
def segment_attention(word_ids_seg, noise_key):
    Ew = Xw[word_ids_seg]
    psi = P[word_ids_seg]

    eps = TOKEN_NOISE * jax.random.normal(noise_key, Ew.shape)
    Eseg = Ew + (psi @ Zc) + eps

    Q = Eseg @ Wq
    K = Eseg @ Wk
    base_logits = (Q @ K.T) / jnp.sqrt(HEAD_DIM)

    S = word_ids_seg.shape[0]
    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    local = dist <= ATTN_LOCAL_WINDOW

    tpl_bias = (psi @ W_psi) @ psi.T
    dir_bias = (psi @ J_pmi) @ psi.T

    idf_tgt = idf_j[word_ids_seg][None, :]

    logits = base_logits + ALPHA_TEMPLATE * tpl_bias + BETA_DIRECTIONAL * dir_bias + LAMBDA_IDF_TARGET * idf_tgt
    logits = jnp.where(local, logits, -jnp.inf)
    logits = logits - 1e9 * jnp.eye(S, dtype=logits.dtype)

    A = jax.nn.softmax(logits, axis=1)
    pair_mass = psi.T @ A @ psi
    return A, base_logits, tpl_bias, dir_bias, pair_mass

# ============================================================
# Sample segments + print
# ============================================================

starts = [random.randint(0, T - SEG_LEN - 1) for _ in range(N_SEGS)]

print("\nCompiling segment attention (one-time JAX cost)...")
dummy = tok_ids[starts[0]:starts[0] + SEG_LEN].copy()
dummy[dummy < 0] = 0
_ = segment_attention(jnp.array(dummy, dtype=jnp.int32), key)[0].block_until_ready()
print("Compilation finished.\n")

template_flow = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float64)
token_links_idf = []

for si, start in enumerate(starts, 1):
    seg = tok_ids[start:start + SEG_LEN].copy()
    seg[seg < 0] = 0
    seg_tokens = [vocab[int(x)] for x in seg]
    seg_j = jnp.array(seg, dtype=jnp.int32)

    key, nk = jax.random.split(key)
    A, baseL, tplB, dirB, pair_mass = segment_attention(seg_j, nk)

    template_flow += np.array(pair_mass)

    A_np = np.array(A)
    base_np = np.array(baseL)
    tpl_np = np.array(tplB)
    dir_np = np.array(dirB)
    idf_seg = idf[seg]

    flat_idf = (A_np * (idf_seg[:, None] * idf_seg[None, :])).ravel()
    top_idf = np.argpartition(flat_idf, -TOP_TOKEN_LINKS)[-TOP_TOKEN_LINKS:]
    top_idf = top_idf[np.argsort(-flat_idf[top_idf])]

    for idx in top_idf:
        i = int(idx // SEG_LEN)
        j = int(idx % SEG_LEN)
        if i == j:
            continue
        wi = int(seg[i]); wj = int(seg[j])
        if SKIP_SELF_LINKS and wi == wj:
            continue
        if idf[wi] < MIN_IDF_TO_PRINT or idf[wj] < MIN_IDF_TO_PRINT:
            continue
        if SKIP_PUNCT_HEAVY and (is_punct_heavy(seg_tokens[i]) or is_punct_heavy(seg_tokens[j])):
            continue

        dist = abs((start + i) - (start + j))
        token_links_idf.append({
            "seg": si,
            "w_i": seg_tokens[i],
            "w_j": seg_tokens[j],
            "score": float(flat_idf[idx]),
            "attn": float(A_np[i, j]),
            "base": float(base_np[i, j]),
            "tpl": float(tpl_np[i, j]),
            "dir": float(dir_np[i, j]),
            "idf_i": float(idf[wi]),
            "idf_j": float(idf[wj]),
            "dist": dist,
        })

print("-" * 80)
print("TEMPLATE→TEMPLATE ATTENTION FLOWS (aggregated over sampled segments)  [SOFT]")
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
    print(f"Template {a:02d} → Template {b:02d}   flow={val:.4f}   J_pmi={J_pmi_np[a,b]:+.3f}   Wpsi={float(W_psi[a,b]):.4f}")

print("\n" + "-" * 80)
print("TOP TOKEN→TOKEN LINKS (IDF-weighted, filtered, ngram embeddings)  [SOFT]")
print("-" * 80)

token_links_idf.sort(key=lambda r: r["score"], reverse=True)
for r in token_links_idf[:TOP_TOKEN_LINKS]:
    near = "LOCAL" if r["dist"] <= ATTN_LOCAL_WINDOW else "LONG"
    print(
        f"[seg {r['seg']}] {near:>4}  {r['w_i']:<14} → {r['w_j']:<14}  "
        f"score={r['score']:.4f} (attn={r['attn']:.4f})  "
        f"base={r['base']:+.3f} tpl={ALPHA_TEMPLATE*r['tpl']:+.3f} dir={BETA_DIRECTIONAL*r['dir']:+.3f}  "
        f"idf=({r['idf_i']:.2f},{r['idf_j']:.2f}) dist={r['dist']}"
    )

print("\nDone.")
