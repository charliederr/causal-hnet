"""
Causal Multi-Head Attention with Hamiltonian Relational Constraints

Phase 1 Implementation based on Causal-HNet paper (Goertzel, Dec 2025)

Key additions over causal_attention_multihead.py:
1. Learnable Hamiltonian relation matrices H_r (Section 3.1, Eq. 5)
2. Binary-consistent pseudo-Boolean energy extension (Section 4.2, Eq. 9)
3. Binarization regularizer F_bin (Section 4.3, Eq. 10)
4. Hamiltonian attention bias replacing simple similarity scores (Section 7, Eq. 13)

The core insight from the paper: relational constraints should be expressed as
energies that are ZERO when satisfied and POSITIVE otherwise. This differs from
similarity-based approaches where higher = better.
"""

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax


# ============================================================
# Utilities (unchanged from original)
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
# Hamiltonian Relation Types
# ============================================================

# Define semantic relation types following the paper's framework.
# Each relation r has a 2x2 matrix H_r and offset k_r such that
# E_r(x,y) = 0 when (x,y) satisfies relation r.

RELATION_NAMES = [
    "IMPLIES",      # x=1 -> y=1 (violation when x=1, y=0)
    "EXCLUDES",     # x=1 -> y=0 (violation when x=1, y=1)
    "REQUIRES",     # y=1 -> x=1 (violation when x=0, y=1)
    "COOCCURS",     # x and y tend to co-occur (low energy when both 1)
    "PRECEDES",     # directional: x precedes y in sequence
    "FOLLOWS",      # directional: x follows y in sequence
]


def get_default_hamiltonian_matrices(n_relations: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize H_r matrices with semantically meaningful defaults.

    For binary states x, y in {0,1}, the quadratic energy is:
        E_r(x,y) = [x y] H_r [x; y]^T + k_r

    We design H_r and k_r so E_r = 0 when the relation is satisfied.

    On {0,1}, since x^2 = x, the energy simplifies to (Eq. 8):
        E_r(x,y) = h11*x + (h12+h21)*x*y + h22*y + k_r

    Returns:
        H_r: (n_relations, 2, 2) - Hamiltonian matrices
        k_r: (n_relations,) - offset constants
    """
    H_r = np.zeros((n_relations, 2, 2), dtype=np.float32)
    k_r = np.zeros(n_relations, dtype=np.float32)

    # IMPLIES: x=1 -> y=1
    # Violation when x=1, y=0. Energy = x(1-y) = x - xy
    # h11=1, h12+h21=-1, h22=0, k=0
    # Check: (1,1)->0, (1,0)->1, (0,1)->0, (0,0)->0
    if n_relations > 0:
        H_r[0] = [[1, -0.5], [-0.5, 0]]
        k_r[0] = 0

    # EXCLUDES: x=1 -> y=0 (mutual exclusion)
    # Violation when x=1, y=1. Energy = xy
    # h11=0, h12+h21=1, h22=0, k=0
    # Check: (1,1)->1, (1,0)->0, (0,1)->0, (0,0)->0
    if n_relations > 1:
        H_r[1] = [[0, 0.5], [0.5, 0]]
        k_r[1] = 0

    # REQUIRES: y=1 -> x=1 (reverse implication)
    # Violation when x=0, y=1. Energy = y(1-x) = y - xy
    # h11=0, h12+h21=-1, h22=1, k=0
    if n_relations > 2:
        H_r[2] = [[0, -0.5], [-0.5, 1]]
        k_r[2] = 0

    # COOCCURS: prefer both on or both off
    # Energy = (x-y)^2 = x - 2xy + y (using x^2=x)
    # Low when x=y, high when x!=y
    if n_relations > 3:
        H_r[3] = [[1, -1], [-1, 1]]
        k_r[3] = 0

    # PRECEDES: directional affinity (asymmetric)
    # Lower energy when x=1 precedes y=1
    # We make this slightly favor the (1,1) case with asymmetric weights
    if n_relations > 4:
        H_r[4] = [[0.2, -0.3], [-0.7, 0.2]]
        k_r[4] = 0

    # FOLLOWS: reverse of PRECEDES
    if n_relations > 5:
        H_r[5] = [[0.2, -0.7], [-0.3, 0.2]]
        k_r[5] = 0

    return H_r, k_r


# ============================================================
# Model Parameters with Hamiltonian Extensions
# ============================================================


class ModelParams(NamedTuple):
    """
    Learnable parameters for the causal attention model.

    Extended with Hamiltonian relation parameters per the paper.
    """
    # Original parameters
    Zc: jnp.ndarray      # Template prototypes: (n_clusters, emb_dim)
    Wq: jnp.ndarray      # Query projections: (n_heads, emb_dim, head_dim)
    Wk: jnp.ndarray      # Key projections: (n_heads, emb_dim, head_dim)
    Wv: jnp.ndarray      # Value projections: (n_heads, emb_dim, head_dim)
    Wo: jnp.ndarray      # Output projection: (n_heads * head_dim, emb_dim)

    # NEW: Hamiltonian relation parameters (Section 3.1)
    H_r: jnp.ndarray     # Relation matrices: (n_relations, 2, 2)
    k_r: jnp.ndarray     # Relation offsets: (n_relations,)

    # NEW: Relation mixture predictor weights (Section 5.1)
    # Maps template pair features to relation type weights
    W_rel: jnp.ndarray   # (2 * emb_dim, n_relations) for predicting w_ij,r


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
    n_relations: int = 6,
) -> ModelParams:
    """Initialize model parameters including Hamiltonian relations."""
    keys = jax.random.split(key, 7)

    # Original parameters
    Zc = jax.random.normal(keys[0], (n_clusters, emb_dim)) * 0.5
    Wq = jax.random.normal(keys[1], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wk = jax.random.normal(keys[2], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wv = jax.random.normal(keys[3], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wo = jax.random.normal(keys[4], (n_heads * head_dim, emb_dim)) / math.sqrt(
        n_heads * head_dim
    )

    # Initialize Hamiltonian matrices with meaningful defaults
    H_r_np, k_r_np = get_default_hamiltonian_matrices(n_relations)
    H_r = jnp.array(H_r_np)
    k_r = jnp.array(k_r_np)

    # Relation mixture predictor: maps concatenated template embeddings to relation weights
    W_rel = jax.random.normal(keys[5], (2 * emb_dim, n_relations)) / math.sqrt(2 * emb_dim)

    return ModelParams(Zc=Zc, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, H_r=H_r, k_r=k_r, W_rel=W_rel)


# ============================================================
# Hamiltonian Energy Functions (Paper Sections 3-4)
# ============================================================


def relaxed_pair_energy(
    p_i: jnp.ndarray,
    p_j: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
) -> jnp.ndarray:
    """
    Binary-consistent (pseudo-Boolean) energy extension.

    From Section 4.2, Equation 9:
        E_r(p,q) = h11*p + (h12+h21)*p*q + h22*q + k_r

    This equals E[E_r(X,Y)] under independent Bernoulli variables
    X ~ Bern(p) and Y ~ Bern(q), which is a natural relaxation
    for logic-like energies.

    Args:
        p_i: Soft state for node i, shape (n_relations,) or scalar in [0,1]
        p_j: Soft state for node j, shape (n_relations,) or scalar in [0,1]
        H_r: Hamiltonian matrices, shape (n_relations, 2, 2)
        k_r: Offset constants, shape (n_relations,)

    Returns:
        Energy values per relation type, shape (n_relations,)
    """
    h11 = H_r[:, 0, 0]  # (n_relations,)
    h12 = H_r[:, 0, 1]
    h21 = H_r[:, 1, 0]
    h22 = H_r[:, 1, 1]

    # Equation 9: linear + bilinear terms (no squares since x^2 = x on binary)
    energy = h11 * p_i + (h12 + h21) * p_i * p_j + h22 * p_j + k_r

    return energy


def relaxed_pair_energy_batched(
    p: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute pairwise Hamiltonian energies for all token pairs.

    Args:
        p: Soft node states, shape (S,) where S is sequence length
           Each p_i in [0,1] represents P(node i is "on")
        H_r: Hamiltonian matrices, shape (n_relations, 2, 2)
        k_r: Offset constants, shape (n_relations,)

    Returns:
        Energy matrix, shape (S, S, n_relations)
        Entry [i,j,r] = E_r(p_i, p_j)
    """
    S = p.shape[0]
    n_relations = H_r.shape[0]

    h11 = H_r[:, 0, 0]  # (n_relations,)
    h12 = H_r[:, 0, 1]
    h21 = H_r[:, 1, 0]
    h22 = H_r[:, 1, 1]

    # Broadcast to compute all pairs
    p_i = p[:, None, None]  # (S, 1, 1)
    p_j = p[None, :, None]  # (1, S, 1)

    # (S, S, n_relations)
    energy = (
        h11[None, None, :] * p_i +
        (h12 + h21)[None, None, :] * p_i * p_j +
        h22[None, None, :] * p_j +
        k_r[None, None, :]
    )

    return energy


def binarization_regularizer(p: jnp.ndarray, beta: float = 0.1) -> jnp.ndarray:
    """
    Binarization regularizer from Section 4.3, Equation 10.

        F_bin(p) = beta * sum_i p_i * (1 - p_i)

    This is minimized when p_i in {0, 1}, encouraging near-binary
    soft states which better match the discrete relational semantics.

    Args:
        p: Soft node states, shape (S,) or (S, C) for templates
        beta: Regularization strength

    Returns:
        Scalar regularization loss
    """
    return beta * jnp.sum(p * (1 - p))


# ============================================================
# Template Functions (extended with Hamiltonian integration)
# ============================================================


def compute_soft_templates(
    Xw: jnp.ndarray, Zc: jnp.ndarray, tau: float, gamma_ctx: float = 0.0
) -> jnp.ndarray:
    """Compute soft template assignments P(c|w) via softmax."""
    scores = Xw @ Zc.T  # (V, C)
    if gamma_ctx > 0:
        ctx = jnp.mean(Xw, axis=0)
        ctx_scores = ctx @ Zc.T
        scores = scores + gamma_ctx * ctx_scores[None, :]
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

    J_pmi = J_pmi - jnp.mean(J_pmi)
    J_pmi = J_pmi / (jnp.std(J_pmi) + 1e-6)

    return J_pmi


def compute_node_soft_states(
    psi: jnp.ndarray,
    Zc: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Derive soft node states p_i from template assignments.

    The paper uses p = sigma(u/T) where u are logits from latents.
    Here we derive a scalar "activation" state per token from its
    template assignment, representing how "active" or "salient" the token is.

    We compute this as the entropy-weighted norm of the template assignment,
    giving higher values to tokens with confident, high-magnitude assignments.

    Args:
        psi: Template assignments, shape (S, C)
        Zc: Template prototypes, shape (C, D)
        temperature: Softmax temperature for state derivation

    Returns:
        Soft node states p, shape (S,) with values in [0, 1]
    """
    # Compute "salience" as weighted template magnitude
    template_norms = jnp.linalg.norm(Zc, axis=1)  # (C,)
    weighted_norm = jnp.sum(psi * template_norms[None, :], axis=1)  # (S,)

    # Normalize to [0, 1] via sigmoid
    p = jax.nn.sigmoid(weighted_norm / temperature)

    return p


def predict_relation_weights(
    psi_i: jnp.ndarray,
    psi_j: jnp.ndarray,
    Zc: jnp.ndarray,
    W_rel: jnp.ndarray,
) -> jnp.ndarray:
    """
    Predict relation-type mixture weights w_ij,r for token pairs.

    From Section 5.1: For each edge (i,j), predict mixture weights
    w_ij,r >= 0 with sum_r w_ij,r = 1.

    Args:
        psi_i: Template assignments for source tokens, shape (S, C)
        psi_j: Template assignments for target tokens, shape (S, C)
        Zc: Template prototypes, shape (C, D)
        W_rel: Relation predictor weights, shape (2*D, n_relations)

    Returns:
        Relation weights, shape (S, S, n_relations)
    """
    # Get template-weighted embeddings
    emb_i = psi_i @ Zc  # (S, D)
    emb_j = psi_j @ Zc  # (S, D)

    S = emb_i.shape[0]
    D = emb_i.shape[1]

    # Concatenate all pairs: (S, S, 2*D)
    emb_i_exp = jnp.broadcast_to(emb_i[:, None, :], (S, S, D))
    emb_j_exp = jnp.broadcast_to(emb_j[None, :, :], (S, S, D))
    pair_features = jnp.concatenate([emb_i_exp, emb_j_exp], axis=-1)

    # Project to relation logits and softmax
    logits = pair_features @ W_rel  # (S, S, n_relations)
    weights = jax.nn.softmax(logits, axis=-1)

    return weights


# ============================================================
# Hamiltonian Attention Bias (Paper Section 7, Eq. 13)
# ============================================================


def compute_hamiltonian_attention_bias(
    p: jnp.ndarray,
    w_rel: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Compute Hamiltonian attention bias from Equation 13.

        l_ij <- l_ij - gamma * E_pair(p_i, p_j)

    where E_pair is the weighted sum of relation energies:
        E_pair(p_i, p_j) = sum_r w_ij,r * E_r(p_i, p_j)

    This SUBTRACTS energy from logits, so low energy (satisfied constraints)
    leads to HIGHER attention, and high energy (violations) leads to
    LOWER attention.

    Args:
        p: Soft node states, shape (S,)
        w_rel: Relation mixture weights, shape (S, S, n_relations)
        H_r: Hamiltonian matrices, shape (n_relations, 2, 2)
        k_r: Relation offsets, shape (n_relations,)
        gamma: Bias strength

    Returns:
        Attention bias matrix, shape (S, S)
    """
    # Compute energies for all pairs and relations: (S, S, n_relations)
    E_all = relaxed_pair_energy_batched(p, H_r, k_r)

    # Weight by relation mixture: sum over relations
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)  # (S, S)

    # Subtract from logits (low energy = high attention)
    bias = -gamma * E_weighted

    return bias


# ============================================================
# Multi-Head Attention with Hamiltonian Biases
# ============================================================


def multihead_attention_hamiltonian(
    Eseg: jnp.ndarray,          # (S, D)
    psi: jnp.ndarray,           # (S, C)
    p: jnp.ndarray,             # (S,) soft node states
    w_rel: jnp.ndarray,         # (S, S, n_relations)
    W_psi: jnp.ndarray,         # (C, C) template compatibility (kept for comparison)
    J_pmi: jnp.ndarray,         # (C, C) directional coupling (kept for comparison)
    idf_seg: jnp.ndarray,       # (S,)
    params: ModelParams,
    head_dim: int,
    local_window: int,
    gamma_hamiltonian: float,   # NEW: Hamiltonian bias strength
    alpha_template: float,      # Original template bias (can reduce/disable)
    beta_directional: float,    # Original directional bias (can reduce/disable)
    lambda_idf: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Multi-head attention with Hamiltonian relational biases.

    This replaces the simple similarity-based biases with proper
    Hamiltonian energies that are zero when constraints are satisfied.

    Returns:
        output: (S, D) attention output
        A_combined: (S, S) combined attention weights
        diagnostics: dict with per-head patterns and energy info
    """
    S, D = Eseg.shape
    n_heads = params.Wq.shape[0]

    # Position indices for local attention mask
    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    local_mask = dist <= local_window

    # Original biases (kept for comparison, can be weighted down)
    tpl_bias = (psi @ W_psi) @ psi.T
    dir_bias = (psi @ J_pmi) @ psi.T
    idf_tgt = idf_seg[None, :]

    # NEW: Hamiltonian attention bias (Eq. 13)
    hamiltonian_bias = compute_hamiltonian_attention_bias(
        p, w_rel, params.H_r, params.k_r, gamma=gamma_hamiltonian
    )

    def single_head_attention(head_idx):
        Wq_h = params.Wq[head_idx]
        Wk_h = params.Wk[head_idx]
        Wv_h = params.Wv[head_idx]

        Q = Eseg @ Wq_h  # (S, head_dim)
        K = Eseg @ Wk_h
        V = Eseg @ Wv_h

        # Base attention logits
        base_logits = (Q @ K.T) / jnp.sqrt(head_dim)

        # Combined biases: Hamiltonian + legacy
        logits = (
            base_logits
            + hamiltonian_bias                    # NEW: Hamiltonian energy bias
            + alpha_template * tpl_bias           # Legacy: template similarity
            + beta_directional * dir_bias         # Legacy: directional coupling
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

    # Compute energy statistics for diagnostics
    E_all = relaxed_pair_energy_batched(p, params.H_r, params.k_r)
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)

    diagnostics = {
        "head_attns": jnp.stack(head_attns, axis=0),
        "head_base_logits": jnp.stack(head_base_logits, axis=0),
        "tpl_bias": tpl_bias,
        "dir_bias": dir_bias,
        "hamiltonian_bias": hamiltonian_bias,
        "E_weighted": E_weighted,
        "E_per_relation": E_all,
        "w_rel": w_rel,
    }

    return output, A_combined, diagnostics


# ============================================================
# Loss Functions
# ============================================================


def template_entropy(P: jnp.ndarray) -> jnp.ndarray:
    """Compute mean entropy of template assignments."""
    return -jnp.mean(jnp.sum(P * jnp.log(P + 1e-12), axis=1))


def template_clustering_loss_hamiltonian(
    Zc: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
    Xw: jnp.ndarray,
    edges_i: jnp.ndarray,
    edges_j: jnp.ndarray,
    edges_w: jnp.ndarray,
    tau: float,
    entropy_weight: float,
    binarization_weight: float,
) -> jnp.ndarray:
    """
    Loss for learning template prototypes with Hamiltonian regularization.

    Extended from original to include:
    - Binarization regularizer (Eq. 10) on soft template assignments
    - Hamiltonian energy term encouraging low-energy configurations

    Args:
        Zc: Template prototypes
        H_r: Hamiltonian relation matrices
        k_r: Relation offsets
        Xw: Word embeddings
        edges_*: Co-occurrence edge data
        tau: Softmax temperature
        entropy_weight: Weight for entropy regularization
        binarization_weight: Weight for binarization regularizer (beta in Eq. 10)
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

    # NEW: Binarization regularizer (Eq. 10)
    # Encourage template assignments to be near 0 or 1
    bin_loss = binarization_regularizer(P, beta=binarization_weight)

    return edge_loss + ent_loss + bin_loss


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Causal Attention with Hamiltonian Relational Constraints (Phase 1)"
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
    ap.add_argument("--binarization_weight", type=float, default=0.05,
                    help="Weight for binarization regularizer (Eq. 10)")

    # Hamiltonian relations
    ap.add_argument("--n_relations", type=int, default=6,
                    help="Number of Hamiltonian relation types")
    ap.add_argument("--gamma_hamiltonian", type=float, default=0.5,
                    help="Strength of Hamiltonian attention bias")

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

    # Legacy bias weights (can reduce to emphasize Hamiltonian)
    ap.add_argument("--alpha_template", type=float, default=0.5,
                    help="Legacy template similarity bias (reduced from 1.0)")
    ap.add_argument("--beta_directional", type=float, default=0.4,
                    help="Legacy directional coupling bias (reduced from 0.75)")
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
    print("CAUSAL ATTENTION WITH HAMILTONIAN RELATIONAL CONSTRAINTS")
    print("Phase 1: Binary-Consistent Energy Extension")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"\nHamiltonian parameters:")
    print(f"  Relations: {args.n_relations} ({', '.join(RELATION_NAMES[:args.n_relations])})")
    print(f"  Gamma (bias strength): {args.gamma_hamiltonian}")
    print(f"  Binarization weight: {args.binarization_weight}")

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
    # Initialize parameters with Hamiltonian extensions
    # --------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    params = init_params(
        init_key,
        args.n_clusters,
        args.emb_dim,
        args.n_heads,
        args.head_dim,
        args.n_relations,
    )

    print(f"\nInitialized Hamiltonian relation matrices:")
    for r in range(min(args.n_relations, len(RELATION_NAMES))):
        H = np.array(params.H_r[r])
        print(f"  {RELATION_NAMES[r]}: H_r = [[{H[0,0]:.2f}, {H[0,1]:.2f}], [{H[1,0]:.2f}, {H[1,1]:.2f}]]")

    # --------------------------------------------------------
    # Train template prototypes with Hamiltonian regularization
    # --------------------------------------------------------
    print(f"\nLearning template prototypes ({args.template_steps} steps)...")

    optimizer = optax.adam(args.template_lr)
    opt_state = optimizer.init(params.Zc)

    @jit
    def train_step(Zc, opt_state):
        loss, grads = jax.value_and_grad(template_clustering_loss_hamiltonian)(
            Zc,
            params.H_r,
            params.k_r,
            Xw,
            edges_i,
            edges_j,
            edges_w,
            args.tau,
            args.entropy_weight,
            args.binarization_weight,
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
            bin_loss = float(binarization_regularizer(P, args.binarization_weight))
            print(f"  step {step+1}/{args.template_steps}  loss={float(loss):.4f}  "
                  f"entropy={ent:.3f}  bin_loss={bin_loss:.4f}")

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
    # Sample segments and run Hamiltonian attention
    # --------------------------------------------------------
    starts = [random.randint(0, T - args.seg_len - 1) for _ in range(args.n_segs)]

    print("\n" + "-" * 80)
    print(f"HAMILTONIAN MULTI-HEAD ATTENTION ANALYSIS ({args.n_heads} heads)")
    print("-" * 80)

    template_flow = np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
    per_head_flow = [
        np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
        for _ in range(args.n_heads)
    ]
    all_edges = []

    # Track Hamiltonian energy statistics
    energy_stats = {r: [] for r in range(args.n_relations)}

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start : start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]

        # Get embeddings and template assignments for segment
        Ew = Xw[seg]
        psi = P[seg]
        idf_seg = idf_j[seg]

        # Compute soft node states (NEW)
        p = compute_node_soft_states(psi, Zc, temperature=1.0)

        # Compute relation mixture weights (NEW)
        w_rel = predict_relation_weights(psi, psi, Zc, params.W_rel)

        # Add template contribution and noise
        key, noise_key = jax.random.split(key)
        eps = args.token_noise * jax.random.normal(noise_key, Ew.shape)
        Eseg = Ew + (psi @ Zc) + eps

        # Run Hamiltonian multi-head attention
        output, A_combined, diagnostics = multihead_attention_hamiltonian(
            Eseg,
            psi,
            p,
            w_rel,
            W_psi,
            J_pmi,
            idf_seg,
            params,
            args.head_dim,
            args.local_window,
            args.gamma_hamiltonian,
            args.alpha_template,
            args.beta_directional,
            args.lambda_idf,
        )

        # Collect energy statistics
        E_per_rel = np.array(diagnostics["E_per_relation"])
        for r in range(args.n_relations):
            energy_stats[r].append(float(np.mean(E_per_rel[:, :, r])))

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
    print("HAMILTONIAN ENERGY STATISTICS (per relation type)")
    print("-" * 80)
    for r in range(args.n_relations):
        if r < len(RELATION_NAMES):
            mean_e = np.mean(energy_stats[r])
            std_e = np.std(energy_stats[r])
            print(f"  {RELATION_NAMES[r]:12s}: mean={mean_e:+.4f}  std={std_e:.4f}")

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
        scores = P_np[:, c] * idf
        top_idx = np.argsort(-scores)[:12]
        words = [vocab[i] for i in top_idx if not is_punct_heavy(vocab[i])][:8]
        print(f"Template {c:02d}: {', '.join(words)}")

    print("\n" + "=" * 80)
    print("Phase 1 Implementation Complete")
    print("=" * 80)
    print("Key additions from Causal-HNet paper:")
    print("  - Hamiltonian relation matrices H_r (Eq. 5)")
    print("  - Binary-consistent energy extension (Eq. 9)")
    print("  - Binarization regularizer F_bin (Eq. 10)")
    print("  - Hamiltonian attention bias (Eq. 13)")
    print("\nNext phases to implement:")
    print("  - Phase 2: Soft edge masks and full F_rel (Section 5)")
    print("  - Phase 3: Predictive-coding integration (Section 2.1)")
    print("  - Phase 4: Context-gated modules (Section 2.2)")


if __name__ == "__main__":
    main()
