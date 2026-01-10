"""
Causal Multi-Head Attention with Learnable Templates

This module extends the causal-hnet framework with:
- Multi-head attention where each head learns different template compatibility patterns
- Learnable template prototypes refined via gradient descent
- Per-head causal flow analysis
- Entropy regularization to prevent template collapse

Building on: soft_constraints_hnet_v7.py and soft_constraints_attention_hierarchy_eventlocal_v4.py
"""

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax


# ============================================================
# Utilities
# ============================================================

_punct_re = re.compile(r"^[\W_]+$", re.UNICODE)


def is_punct_heavy(token: str) -> bool:
    if _punct_re.match(token):
        return True
    alnum = sum(ch.isalnum() for ch in token)
    return alnum <= 1 and len(token) <= 4


def normalize_for_ngrams(token: str) -> str:
    t = token.lower()
    t = re.sub(r"^[^a-z0-9]+", "", t)
    t = re.sub(r"[^a-z0-9]+$", "", t)
    return t


def char_ngrams(s: str, nmin: int, nmax: int) -> List[str]:
    if not s:
        return []
    s2 = f"<{s}>"
    out = []
    L = len(s2)
    for n in range(nmin, nmax + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            out.append(s2[i : i + n])
    return out


def stable_hash32(x: str) -> int:
    """FNV-1a hash for deterministic hashing."""
    h = 2166136261
    for ch in x:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ============================================================
# Model Parameters (as a NamedTuple for clean handling)
# ============================================================


class ModelParams(NamedTuple):
    """Learnable parameters for the causal attention model."""

    Zc: jnp.ndarray  # Template prototypes: (n_clusters, emb_dim)
    Wq: jnp.ndarray  # Query projections: (n_heads, emb_dim, head_dim)
    Wk: jnp.ndarray  # Key projections: (n_heads, emb_dim, head_dim)
    Wv: jnp.ndarray  # Value projections: (n_heads, emb_dim, head_dim)
    Wo: jnp.ndarray  # Output projection: (n_heads * head_dim, emb_dim)


# ============================================================
# Core Functions
# ============================================================


def build_word_embeddings(
    vocab: List[str],
    emb_dim: int,
    ngram_min: int,
    ngram_max: int,
    hash_buckets: int,
    ngram_scale: float,
    seed: int,
) -> jnp.ndarray:
    """Build character n-gram hashed word embeddings."""
    rng = np.random.RandomState(seed)
    bucket_vecs = rng.normal(size=(hash_buckets, emb_dim)).astype(np.float32)
    bucket_vecs *= ngram_scale / math.sqrt(emb_dim)

    V = len(vocab)
    Xw_np = np.zeros((V, emb_dim), dtype=np.float32)

    for wid, token in enumerate(vocab):
        normed = normalize_for_ngrams(token)
        ngs = char_ngrams(normed, ngram_min, ngram_max)
        if not ngs:
            Xw_np[wid] = rng.normal(size=(emb_dim,)).astype(np.float32) * 0.01
            continue
        idxs = [stable_hash32(g) % hash_buckets for g in ngs]
        Xw_np[wid] = bucket_vecs[idxs].mean(axis=0)

    return jnp.array(Xw_np)


def build_cooccurrence_edges(
    tok_ids: np.ndarray, V: int, window: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build directional word->word co-occurrence edges."""
    T = len(tok_ids)
    cooc = defaultdict(float)

    for i in range(T):
        wi = tok_ids[i]
        if wi < 0:
            continue
        end = min(i + window, T)
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

    return (
        jnp.array(edges_i, dtype=jnp.int32),
        jnp.array(edges_j, dtype=jnp.int32),
        jnp.array(edges_w, dtype=jnp.float32),
    )


def init_params(
    key: jax.random.PRNGKey,
    n_clusters: int,
    emb_dim: int,
    n_heads: int,
    head_dim: int,
) -> ModelParams:
    """Initialize model parameters."""
    keys = jax.random.split(key, 5)

    Zc = jax.random.normal(keys[0], (n_clusters, emb_dim)) * 0.5
    Wq = jax.random.normal(keys[1], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wk = jax.random.normal(keys[2], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wv = jax.random.normal(keys[3], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wo = jax.random.normal(keys[4], (n_heads * head_dim, emb_dim)) / math.sqrt(
        n_heads * head_dim
    )

    return ModelParams(Zc=Zc, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo)


def compute_soft_templates(
    Xw: jnp.ndarray, Zc: jnp.ndarray, tau: float, gamma_ctx: float = 0.0
) -> jnp.ndarray:
    """Compute soft template assignments P(c|w) via softmax."""
    # Token-template scores
    scores = Xw @ Zc.T  # (V, C)
    # Optional context bias (mean embedding similarity)
    if gamma_ctx > 0:
        ctx = jnp.mean(Xw, axis=0)
        ctx_scores = ctx @ Zc.T
        scores = scores + gamma_ctx * ctx_scores[None, :]
    # Softmax with temperature
    P = jax.nn.softmax(scores / tau, axis=1)
    return P


def compute_template_compatibility(Zc: jnp.ndarray) -> jnp.ndarray:
    """Compute template compatibility matrix W_psi from prototypes."""
    Zc_norm = Zc / (jnp.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
    return Zc_norm @ Zc_norm.T


def compute_directional_coupling(
    P: jnp.ndarray,
    edges_i: jnp.ndarray,
    edges_j: jnp.ndarray,
    edges_w: jnp.ndarray,
) -> jnp.ndarray:
    """Compute cluster-to-cluster directional coupling matrix J_pmi."""
    Pi = P[edges_i]
    Pj = P[edges_j]
    Pi_w = Pi * edges_w[:, None]

    J_mass = Pi_w.T @ Pj
    J_row = J_mass / (jnp.sum(J_mass, axis=1, keepdims=True) + 1e-12)

    V = P.shape[0]
    pi = jnp.sum(P, axis=0) / float(V)
    base = pi[:, None] * pi[None, :]
    J_pmi = jnp.log((J_row + 1e-12) / (base + 1e-12))

    # Normalize
    J_pmi = J_pmi - jnp.mean(J_pmi)
    J_pmi = J_pmi / (jnp.std(J_pmi) + 1e-6)

    return J_pmi


def multihead_attention(
    Eseg: jnp.ndarray,  # (S, D)
    psi: jnp.ndarray,  # (S, C)
    W_psi: jnp.ndarray,  # (C, C)
    J_pmi: jnp.ndarray,  # (C, C)
    idf_seg: jnp.ndarray,  # (S,)
    params: ModelParams,
    head_dim: int,
    local_window: int,
    alpha_template: float,
    beta_directional: float,
    lambda_idf: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Multi-head attention with template biases.

    Returns:
        output: (S, D) attention output
        A_combined: (S, S) combined attention weights
        diagnostics: dict with per-head attention patterns
    """
    S, D = Eseg.shape
    n_heads = params.Wq.shape[0]

    # Position indices for local attention mask
    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    local_mask = dist <= local_window

    # Template biases (shared across heads)
    tpl_bias = (psi @ W_psi) @ psi.T
    dir_bias = (psi @ J_pmi) @ psi.T
    idf_tgt = idf_seg[None, :]

    def single_head_attention(head_idx):
        Wq_h = params.Wq[head_idx]
        Wk_h = params.Wk[head_idx]
        Wv_h = params.Wv[head_idx]

        Q = Eseg @ Wq_h  # (S, head_dim)
        K = Eseg @ Wk_h
        V = Eseg @ Wv_h

        # Base attention logits
        base_logits = (Q @ K.T) / jnp.sqrt(head_dim)

        # Add template biases
        logits = (
            base_logits
            + alpha_template * tpl_bias
            + beta_directional * dir_bias
            + lambda_idf * idf_tgt
        )

        # Apply local mask and remove self-attention
        logits = jnp.where(local_mask, logits, -1e9)
        logits = logits - 1e9 * jnp.eye(S, dtype=logits.dtype)

        # Softmax
        A = jax.nn.softmax(logits, axis=1)

        # Attend to values
        out = A @ V  # (S, head_dim)

        return out, A, base_logits

    # Run all heads
    head_outputs = []
    head_attns = []
    head_base_logits = []

    for h in range(n_heads):
        out_h, A_h, base_h = single_head_attention(h)
        head_outputs.append(out_h)
        head_attns.append(A_h)
        head_base_logits.append(base_h)

    # Concatenate head outputs
    concat_out = jnp.concatenate(head_outputs, axis=1)  # (S, n_heads * head_dim)

    # Project back to embedding dimension
    output = concat_out @ params.Wo  # (S, D)

    # Combined attention (average across heads)
    A_combined = jnp.stack(head_attns, axis=0).mean(axis=0)

    diagnostics = {
        "head_attns": jnp.stack(head_attns, axis=0),
        "head_base_logits": jnp.stack(head_base_logits, axis=0),
        "tpl_bias": tpl_bias,
        "dir_bias": dir_bias,
    }

    return output, A_combined, diagnostics


def template_entropy(P: jnp.ndarray) -> jnp.ndarray:
    """Compute mean entropy of template assignments."""
    return -jnp.mean(jnp.sum(P * jnp.log(P + 1e-12), axis=1))


def template_clustering_loss(
    Zc: jnp.ndarray,
    Xw: jnp.ndarray,
    edges_i: jnp.ndarray,
    edges_j: jnp.ndarray,
    edges_w: jnp.ndarray,
    tau: float,
    entropy_weight: float,
) -> jnp.ndarray:
    """
    Loss for learning template prototypes.

    Encourages:
    - High co-occurrence edge weight between tokens assigned to compatible templates
    - Entropy regularization to prevent collapse
    """
    P = compute_soft_templates(Xw, Zc, tau)

    # Edge agreement loss: templates should predict co-occurrence
    Pi = P[edges_i]
    Pj = P[edges_j]
    agreement = jnp.sum(Pi * Pj, axis=1)  # template overlap
    edge_loss = -jnp.sum(edges_w * agreement)

    # Entropy regularization (prevent collapse)
    ent = template_entropy(P)
    ent_loss = -entropy_weight * ent

    return edge_loss + ent_loss


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Causal Multi-Head Attention with Learnable Templates"
    )

    # Data
    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--cooc_window", type=int, default=6)

    # Templates
    ap.add_argument("--n_clusters", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.5, help="Softmax temperature")
    ap.add_argument("--template_lr", type=float, default=0.01)
    ap.add_argument("--template_steps", type=int, default=100)
    ap.add_argument("--entropy_weight", type=float, default=0.1)

    # Embeddings
    ap.add_argument("--emb_dim", type=int, default=96)
    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--hash_buckets", type=int, default=2**18)
    ap.add_argument("--ngram_scale", type=float, default=0.2)

    # Attention
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--seg_len", type=int, default=256)
    ap.add_argument("--n_segs", type=int, default=6)
    ap.add_argument("--local_window", type=int, default=64)
    ap.add_argument("--token_noise", type=float, default=0.01)

    # Bias weights
    ap.add_argument("--alpha_template", type=float, default=1.0)
    ap.add_argument("--beta_directional", type=float, default=0.75)
    ap.add_argument("--lambda_idf", type=float, default=0.15)

    # Output
    ap.add_argument("--top_edges", type=int, default=50)
    ap.add_argument("--top_template_flows", type=int, default=16)
    ap.add_argument("--min_idf_print", type=float, default=1.8)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("CAUSAL MULTI-HEAD ATTENTION WITH LEARNABLE TEMPLATES")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # --------------------------------------------------------
    # Load text and build vocabulary
    # --------------------------------------------------------
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().lower()

    raw_tokens = text.split()
    T = len(raw_tokens)
    print(f"\nTotal tokens: {T}")

    freq = Counter(raw_tokens)
    vocab = [w for w, c in freq.items() if c >= args.min_freq]
    vocab = vocab[: args.max_vocab]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"Vocabulary size: {V}")

    tok_ids = np.full((T,), -1, dtype=np.int32)
    for i, w in enumerate(raw_tokens):
        tok_ids[i] = word_to_id.get(w, -1)

    # IDF
    counts_vocab = np.zeros(V, dtype=np.int64)
    for tid in tok_ids:
        if tid >= 0:
            counts_vocab[tid] += 1
    N = counts_vocab.sum()
    idf = np.log((N + 1.0) / (counts_vocab + 1.0)) + 1.0
    idf = idf.astype(np.float32)
    idf_j = jnp.array(idf)

    # --------------------------------------------------------
    # Build embeddings and co-occurrence edges
    # --------------------------------------------------------
    print("\nBuilding word embeddings...")
    Xw = build_word_embeddings(
        vocab,
        args.emb_dim,
        args.ngram_min,
        args.ngram_max,
        args.hash_buckets,
        args.ngram_scale,
        args.seed,
    )

    print("Building co-occurrence edges...")
    edges_i, edges_j, edges_w = build_cooccurrence_edges(
        tok_ids, V, args.cooc_window
    )
    print(f"Directional edges: {len(edges_i)}")

    # --------------------------------------------------------
    # Initialize and train template prototypes
    # --------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    params = init_params(
        init_key, args.n_clusters, args.emb_dim, args.n_heads, args.head_dim
    )

    print(f"\nLearning template prototypes ({args.template_steps} steps)...")

    # Optimizer for templates only
    optimizer = optax.adam(args.template_lr)
    opt_state = optimizer.init(params.Zc)

    @jit
    def train_step(Zc, opt_state):
        loss, grads = jax.value_and_grad(template_clustering_loss)(
            Zc, Xw, edges_i, edges_j, edges_w, args.tau, args.entropy_weight
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        Zc = optax.apply_updates(Zc, updates)
        return Zc, opt_state, loss

    Zc = params.Zc
    for step in range(args.template_steps):
        Zc, opt_state, loss = train_step(Zc, opt_state)
        if (step + 1) % 20 == 0 or step == 0:
            P = compute_soft_templates(Xw, Zc, args.tau)
            ent = float(template_entropy(P))
            print(f"  step {step+1}/{args.template_steps}  loss={float(loss):.4f}  entropy={ent:.3f}")

    # Update params with learned templates
    params = params._replace(Zc=Zc)

    # --------------------------------------------------------
    # Compute template matrices
    # --------------------------------------------------------
    P = compute_soft_templates(Xw, Zc, args.tau)
    W_psi = compute_template_compatibility(Zc)
    J_pmi = compute_directional_coupling(P, edges_i, edges_j, edges_w)

    J_pmi_np = np.array(J_pmi)

    print("\n" + "-" * 80)
    print("TOP CLUSTER->CLUSTER DIRECTIONAL COUPLINGS (J_pmi)")
    print("-" * 80)
    pairs = []
    for a in range(args.n_clusters):
        for b in range(args.n_clusters):
            if a != b:
                pairs.append((J_pmi_np[a, b], a, b))
    pairs.sort(reverse=True)
    for val, a, b in pairs[: args.top_template_flows]:
        print(f"c{a:02d} -> c{b:02d}   J_pmi={val:.3f}")

    # --------------------------------------------------------
    # Sample segments and run multi-head attention
    # --------------------------------------------------------
    starts = [random.randint(0, T - args.seg_len - 1) for _ in range(args.n_segs)]

    print("\n" + "-" * 80)
    print(f"MULTI-HEAD ATTENTION ANALYSIS ({args.n_heads} heads)")
    print("-" * 80)

    template_flow = np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
    per_head_flow = [
        np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
        for _ in range(args.n_heads)
    ]
    all_edges = []

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start : start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]

        # Get embeddings and template assignments for segment
        Ew = Xw[seg]
        psi = P[seg]
        idf_seg = idf_j[seg]

        # Add template contribution and noise
        key, noise_key = jax.random.split(key)
        eps = args.token_noise * jax.random.normal(noise_key, Ew.shape)
        Eseg = Ew + (psi @ Zc) + eps

        # Run multi-head attention
        output, A_combined, diagnostics = multihead_attention(
            Eseg,
            psi,
            W_psi,
            J_pmi,
            idf_seg,
            params,
            args.head_dim,
            args.local_window,
            args.alpha_template,
            args.beta_directional,
            args.lambda_idf,
        )

        # Accumulate template flows
        A_np = np.array(A_combined)
        psi_np = np.array(psi)
        pair_mass = psi_np.T @ A_np @ psi_np
        template_flow += pair_mass

        # Per-head flows
        head_attns = np.array(diagnostics["head_attns"])
        for h in range(args.n_heads):
            per_head_flow[h] += psi_np.T @ head_attns[h] @ psi_np

        # Collect top edges
        flat_idf = (A_np * (idf[seg][:, None] * idf[seg][None, :])).ravel()
        top_idx = np.argpartition(flat_idf, -20)[-20:]
        top_idx = top_idx[np.argsort(-flat_idf[top_idx])]

        for idx in top_idx:
            i = int(idx // args.seg_len)
            j = int(idx % args.seg_len)
            if i == j:
                continue
            wi = int(seg[i])
            wj = int(seg[j])
            if idf[wi] < args.min_idf_print or idf[wj] < args.min_idf_print:
                continue
            if is_punct_heavy(seg_words[i]) or is_punct_heavy(seg_words[j]):
                continue

            all_edges.append(
                {
                    "seg": seg_i,
                    "w_i": seg_words[i],
                    "w_j": seg_words[j],
                    "score": float(flat_idf[idx]),
                    "attn": float(A_np[i, j]),
                    "dist": abs(i - j),
                    "idf_i": float(idf[wi]),
                    "idf_j": float(idf[wj]),
                }
            )

    template_flow /= max(1, args.n_segs)
    for h in range(args.n_heads):
        per_head_flow[h] /= max(1, args.n_segs)

    # --------------------------------------------------------
    # Output results
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEMPLATE->TEMPLATE ATTENTION FLOWS (combined)")
    print("-" * 80)
    pairs = []
    for a in range(args.n_clusters):
        for b in range(args.n_clusters):
            if a != b:
                pairs.append((template_flow[a, b], a, b))
    pairs.sort(reverse=True)
    for val, a, b in pairs[: args.top_template_flows]:
        print(
            f"Template {a:02d} -> Template {b:02d}   "
            f"flow={val:.4f}   J_pmi={J_pmi_np[a,b]:+.3f}   W_psi={float(W_psi[a,b]):.3f}"
        )

    # Per-head analysis
    for h in range(args.n_heads):
        print(f"\n-- Head {h} top flows --")
        pairs = []
        for a in range(args.n_clusters):
            for b in range(args.n_clusters):
                if a != b:
                    pairs.append((per_head_flow[h][a, b], a, b))
        pairs.sort(reverse=True)
        for val, a, b in pairs[:8]:
            print(f"  t{a:02d} -> t{b:02d}   flow={val:.4f}")

    print("\n" + "-" * 80)
    print(f"TOP {args.top_edges} TOKEN->TOKEN LINKS (IDF-weighted)")
    print("-" * 80)
    all_edges.sort(key=lambda r: r["score"], reverse=True)
    for r in all_edges[: args.top_edges]:
        near = "LOCAL" if r["dist"] <= args.local_window else "LONG"
        print(
            f"[seg {r['seg']}] {near:>5}  {r['w_i']:<14} -> {r['w_j']:<14}  "
            f"score={r['score']:.4f}  attn={r['attn']:.4f}  "
            f"idf=({r['idf_i']:.2f},{r['idf_j']:.2f})  dist={r['dist']}"
        )

    # --------------------------------------------------------
    # Template word associations
    # --------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEMPLATE WORD ASSOCIATIONS (top words per template)")
    print("-" * 80)

    P_np = np.array(P)
    for c in range(args.n_clusters):
        scores = P_np[:, c] * idf  # weight by IDF
        top_idx = np.argsort(-scores)[:12]
        words = [vocab[i] for i in top_idx if not is_punct_heavy(vocab[i])][:8]
        print(f"Template {c:02d}: {', '.join(words)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
