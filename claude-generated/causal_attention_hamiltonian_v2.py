"""
Causal Multi-Head Attention with Hamiltonian Relational Constraints

Phase 2 Implementation based on Causal-HNet paper (Goertzel, Dec 2025)

Building on Phase 1, this adds:
1. Soft edge masks m_ij(u) ∈ [0,1] (Section 5.1)
2. Full relational energy F_rel (Section 5.2, Eq. 11)
3. Sparse candidate edge selection for O(Tk) instead of O(T²)
4. Combined objective F_total = F_PC + λF_rel + F_bin (Section 6, Eq. 12)

Key insight from Section 5: "Building a graph by hard thresholding edges can
create discontinuous energies. Instead, fix a candidate edge set and learn
soft edge weights and edge-type mixtures."
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
# Hamiltonian Relation Types
# ============================================================

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

    On {0,1}, since x^2 = x, the energy simplifies to (Eq. 8):
        E_r(x,y) = h11*x + (h12+h21)*x*y + h22*y + k_r
    """
    H_r = np.zeros((n_relations, 2, 2), dtype=np.float32)
    k_r = np.zeros(n_relations, dtype=np.float32)

    # IMPLIES: x=1 -> y=1, violation when x=1, y=0
    if n_relations > 0:
        H_r[0] = [[1, -0.5], [-0.5, 0]]
        k_r[0] = 0

    # EXCLUDES: x=1 -> y=0 (mutual exclusion)
    if n_relations > 1:
        H_r[1] = [[0, 0.5], [0.5, 0]]
        k_r[1] = 0

    # REQUIRES: y=1 -> x=1 (reverse implication)
    if n_relations > 2:
        H_r[2] = [[0, -0.5], [-0.5, 1]]
        k_r[2] = 0

    # COOCCURS: prefer both on or both off
    if n_relations > 3:
        H_r[3] = [[1, -1], [-1, 1]]
        k_r[3] = 0

    # PRECEDES: directional affinity (asymmetric)
    if n_relations > 4:
        H_r[4] = [[0.2, -0.3], [-0.7, 0.2]]
        k_r[4] = 0

    # FOLLOWS: reverse of PRECEDES
    if n_relations > 5:
        H_r[5] = [[0.2, -0.7], [-0.3, 0.2]]
        k_r[5] = 0

    return H_r, k_r


# ============================================================
# Model Parameters with Phase 2 Extensions
# ============================================================


class ModelParams(NamedTuple):
    """
    Learnable parameters for the causal attention model.

    Phase 2 adds edge mask predictor parameters.
    """
    # Original parameters
    Zc: jnp.ndarray      # Template prototypes: (n_clusters, emb_dim)
    Wq: jnp.ndarray      # Query projections: (n_heads, emb_dim, head_dim)
    Wk: jnp.ndarray      # Key projections: (n_heads, emb_dim, head_dim)
    Wv: jnp.ndarray      # Value projections: (n_heads, emb_dim, head_dim)
    Wo: jnp.ndarray      # Output projection: (n_heads * head_dim, emb_dim)

    # Phase 1: Hamiltonian relation parameters
    H_r: jnp.ndarray     # Relation matrices: (n_relations, 2, 2)
    k_r: jnp.ndarray     # Relation offsets: (n_relations,)
    W_rel: jnp.ndarray   # Relation mixture predictor: (2 * emb_dim, n_relations)

    # Phase 2: Soft edge mask parameters (Section 5.1)
    W_mask_1: jnp.ndarray  # Edge mask MLP layer 1: (2 * emb_dim + n_edge_features, hidden_dim)
    b_mask_1: jnp.ndarray  # Bias for layer 1: (hidden_dim,)
    W_mask_2: jnp.ndarray  # Edge mask MLP layer 2: (hidden_dim, 1)
    b_mask_2: jnp.ndarray  # Bias for layer 2: (1,)


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
    mask_hidden_dim: int = 32,
    n_edge_features: int = 4,
) -> ModelParams:
    """Initialize model parameters including Phase 2 edge mask predictor."""
    keys = jax.random.split(key, 11)

    # Original parameters
    Zc = jax.random.normal(keys[0], (n_clusters, emb_dim)) * 0.5
    Wq = jax.random.normal(keys[1], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wk = jax.random.normal(keys[2], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wv = jax.random.normal(keys[3], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wo = jax.random.normal(keys[4], (n_heads * head_dim, emb_dim)) / math.sqrt(
        n_heads * head_dim
    )

    # Phase 1: Hamiltonian matrices
    H_r_np, k_r_np = get_default_hamiltonian_matrices(n_relations)
    H_r = jnp.array(H_r_np)
    k_r = jnp.array(k_r_np)
    W_rel = jax.random.normal(keys[5], (2 * emb_dim, n_relations)) / math.sqrt(2 * emb_dim)

    # Phase 2: Edge mask MLP
    # Input: concatenated embeddings (2*emb_dim) + edge features (distance, etc.)
    mask_input_dim = 2 * emb_dim + n_edge_features
    W_mask_1 = jax.random.normal(keys[6], (mask_input_dim, mask_hidden_dim)) / math.sqrt(mask_input_dim)
    b_mask_1 = jnp.zeros(mask_hidden_dim)
    W_mask_2 = jax.random.normal(keys[7], (mask_hidden_dim, 1)) / math.sqrt(mask_hidden_dim)
    # Initialize bias slightly negative so masks start near 0.5 after sigmoid
    b_mask_2 = jnp.zeros(1)

    return ModelParams(
        Zc=Zc, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        H_r=H_r, k_r=k_r, W_rel=W_rel,
        W_mask_1=W_mask_1, b_mask_1=b_mask_1,
        W_mask_2=W_mask_2, b_mask_2=b_mask_2,
    )


# ============================================================
# Hamiltonian Energy Functions (Phase 1)
# ============================================================


def relaxed_pair_energy(
    p_i: jnp.ndarray,
    p_j: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
) -> jnp.ndarray:
    """
    Binary-consistent (pseudo-Boolean) energy extension (Eq. 9).

        E_r(p,q) = h11*p + (h12+h21)*p*q + h22*q + k_r
    """
    h11 = H_r[:, 0, 0]
    h12 = H_r[:, 0, 1]
    h21 = H_r[:, 1, 0]
    h22 = H_r[:, 1, 1]

    energy = h11 * p_i + (h12 + h21) * p_i * p_j + h22 * p_j + k_r
    return energy


def relaxed_pair_energy_batched(
    p: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute pairwise Hamiltonian energies for all token pairs.

    Returns:
        Energy matrix, shape (S, S, n_relations)
    """
    h11 = H_r[:, 0, 0]
    h12 = H_r[:, 0, 1]
    h21 = H_r[:, 1, 0]
    h22 = H_r[:, 1, 1]

    p_i = p[:, None, None]
    p_j = p[None, :, None]

    energy = (
        h11[None, None, :] * p_i +
        (h12 + h21)[None, None, :] * p_i * p_j +
        h22[None, None, :] * p_j +
        k_r[None, None, :]
    )

    return energy


def binarization_regularizer(p: jnp.ndarray, beta: float = 0.1) -> jnp.ndarray:
    """
    Binarization regularizer (Eq. 10): F_bin(p) = beta * sum p_i * (1 - p_i)
    """
    return beta * jnp.sum(p * (1 - p))


# ============================================================
# Phase 2: Soft Edge Masks (Section 5.1)
# ============================================================


def compute_edge_features(
    idx_i: jnp.ndarray,
    idx_j: jnp.ndarray,
    seq_len: int,
) -> jnp.ndarray:
    """
    Compute edge features for candidate pairs.

    Features include:
    - Normalized distance
    - Direction (forward/backward)
    - Log distance
    - Position features

    Args:
        idx_i: Source indices, shape (E,)
        idx_j: Target indices, shape (E,)
        seq_len: Sequence length for normalization

    Returns:
        Edge features, shape (E, n_edge_features)
    """
    # Distance features
    dist = idx_j - idx_i  # Can be negative for backward edges
    abs_dist = jnp.abs(dist)

    # Normalized distance [0, 1]
    norm_dist = abs_dist / max(seq_len, 1)

    # Direction: 1 for forward, -1 for backward
    direction = jnp.sign(dist).astype(jnp.float32)

    # Log distance (add 1 to avoid log(0))
    log_dist = jnp.log1p(abs_dist) / jnp.log1p(seq_len)

    # Relative position of source
    rel_pos_i = idx_i / max(seq_len - 1, 1)

    features = jnp.stack([norm_dist, direction, log_dist, rel_pos_i], axis=-1)
    return features


def compute_edge_features_dense(seq_len: int) -> jnp.ndarray:
    """
    Compute edge features for all pairs in a sequence (dense).

    Returns:
        Edge features, shape (S, S, n_edge_features)
    """
    idx = jnp.arange(seq_len)
    idx_i = idx[:, None]  # (S, 1)
    idx_j = idx[None, :]  # (1, S)

    dist = idx_j - idx_i
    abs_dist = jnp.abs(dist)

    norm_dist = abs_dist / max(seq_len, 1)
    direction = jnp.sign(dist).astype(jnp.float32)
    log_dist = jnp.log1p(abs_dist) / jnp.log1p(seq_len)
    rel_pos_i = idx_i / max(seq_len - 1, 1) * jnp.ones((seq_len, seq_len))

    features = jnp.stack([norm_dist, direction, log_dist, rel_pos_i], axis=-1)
    return features


def predict_edge_masks(
    emb_i: jnp.ndarray,
    emb_j: jnp.ndarray,
    edge_features: jnp.ndarray,
    params: ModelParams,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Predict soft edge masks m_ij(u) ∈ [0,1] for candidate edges.

    From Section 5.1: "For each (i,j) ∈ E_0, predict a soft mask m_ij(u) ∈ [0,1]
    (whether the edge matters)"

    Uses a small MLP: concat(emb_i, emb_j, edge_features) -> hidden -> sigmoid

    Args:
        emb_i: Source embeddings, shape (..., D)
        emb_j: Target embeddings, shape (..., D)
        edge_features: Edge features, shape (..., n_edge_features)
        params: Model parameters with mask MLP weights
        temperature: Sigmoid temperature (higher = softer masks)

    Returns:
        Soft edge masks, shape (...) with values in [0, 1]
    """
    # Concatenate inputs
    x = jnp.concatenate([emb_i, emb_j, edge_features], axis=-1)

    # MLP layer 1 with ReLU
    h = jnp.tanh(x @ params.W_mask_1 + params.b_mask_1)

    # MLP layer 2 to scalar
    logits = (h @ params.W_mask_2 + params.b_mask_2).squeeze(-1)

    # Sigmoid with temperature
    masks = jax.nn.sigmoid(logits / temperature)

    return masks


def predict_edge_masks_dense(
    Eseg: jnp.ndarray,
    edge_features: jnp.ndarray,
    params: ModelParams,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Predict soft edge masks for all pairs (dense computation).

    Args:
        Eseg: Token embeddings, shape (S, D)
        edge_features: Edge features, shape (S, S, n_edge_features)
        params: Model parameters
        temperature: Sigmoid temperature

    Returns:
        Edge masks, shape (S, S)
    """
    S, D = Eseg.shape

    # Broadcast embeddings to all pairs
    emb_i = jnp.broadcast_to(Eseg[:, None, :], (S, S, D))
    emb_j = jnp.broadcast_to(Eseg[None, :, :], (S, S, D))

    return predict_edge_masks(emb_i, emb_j, edge_features, params, temperature)


# ============================================================
# Phase 2: Relational Energy F_rel (Section 5.2, Eq. 11)
# ============================================================


def compute_relational_energy(
    p: jnp.ndarray,
    m: jnp.ndarray,
    w_rel: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
    candidate_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute relational energy F_rel from Equation 11.

        F_rel(u) = Σ_{(i,j)∈E_0} m_ij(u) Σ_r w_ij,r(u) Ẽ_r(p_i(u), p_j(u))

    This is the weighted sum of Hamiltonian pair energies, where:
    - m_ij is the soft edge mask (whether edge matters)
    - w_ij,r is the relation type mixture weight
    - Ẽ_r is the relaxed pair energy

    Args:
        p: Soft node states, shape (S,)
        m: Soft edge masks, shape (S, S)
        w_rel: Relation mixture weights, shape (S, S, n_relations)
        H_r: Hamiltonian matrices, shape (n_relations, 2, 2)
        k_r: Relation offsets, shape (n_relations,)
        candidate_mask: Optional binary mask for sparse computation, shape (S, S)

    Returns:
        Scalar relational energy
    """
    # Compute all pair energies: (S, S, n_relations)
    E_all = relaxed_pair_energy_batched(p, H_r, k_r)

    # Weight by relation mixture: (S, S)
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)

    # Apply edge masks
    E_masked = m * E_weighted

    # Apply candidate mask if provided (for sparse computation)
    if candidate_mask is not None:
        E_masked = E_masked * candidate_mask

    # Sum over all edges
    F_rel = jnp.sum(E_masked)

    return F_rel


def compute_relational_energy_sparse(
    p: jnp.ndarray,
    emb: jnp.ndarray,
    candidate_i: jnp.ndarray,
    candidate_j: jnp.ndarray,
    params: ModelParams,
    seq_len: int,
    mask_temperature: float = 1.0,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute relational energy on sparse candidate edges only.

    This achieves O(E) computation instead of O(S²) where E is
    the number of candidate edges.

    Args:
        p: Soft node states, shape (S,)
        emb: Token embeddings, shape (S, D)
        candidate_i: Source indices for candidate edges, shape (E,)
        candidate_j: Target indices for candidate edges, shape (E,)
        params: Model parameters
        seq_len: Sequence length
        mask_temperature: Temperature for edge mask sigmoid

    Returns:
        F_rel: Scalar relational energy
        diagnostics: Dict with intermediate values
    """
    E = len(candidate_i)

    # Get embeddings for candidate pairs
    emb_i = emb[candidate_i]  # (E, D)
    emb_j = emb[candidate_j]  # (E, D)

    # Compute edge features
    edge_features = compute_edge_features(candidate_i, candidate_j, seq_len)

    # Predict edge masks
    m = predict_edge_masks(emb_i, emb_j, edge_features, params, mask_temperature)

    # Predict relation weights
    pair_features = jnp.concatenate([emb_i, emb_j], axis=-1)
    rel_logits = pair_features @ params.W_rel
    w_rel = jax.nn.softmax(rel_logits, axis=-1)  # (E, n_relations)

    # Get soft states for pairs
    p_i = p[candidate_i]  # (E,)
    p_j = p[candidate_j]  # (E,)

    # Compute pair energies for each relation
    E_per_rel = relaxed_pair_energy(p_i[:, None], p_j[:, None], params.H_r, params.k_r)

    # Weight by relation mixture
    E_weighted = jnp.sum(w_rel * E_per_rel, axis=-1)  # (E,)

    # Apply edge masks and sum
    F_rel = jnp.sum(m * E_weighted)

    diagnostics = {
        "edge_masks": m,
        "relation_weights": w_rel,
        "pair_energies": E_weighted,
        "E_per_relation": E_per_rel,
    }

    return F_rel, diagnostics


# ============================================================
# Sparse Candidate Edge Selection (Section 7)
# ============================================================


def select_candidate_edges_local(
    seq_len: int,
    local_window: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select candidate edges based on local window.

    Returns edges within local_window distance (both directions).

    Args:
        seq_len: Sequence length
        local_window: Maximum distance for candidate edges

    Returns:
        candidate_i: Source indices, shape (E,)
        candidate_j: Target indices, shape (E,)
    """
    idx = np.arange(seq_len)
    candidate_i = []
    candidate_j = []

    for i in range(seq_len):
        for j in range(max(0, i - local_window), min(seq_len, i + local_window + 1)):
            if i != j:  # No self-loops
                candidate_i.append(i)
                candidate_j.append(j)

    return jnp.array(candidate_i, dtype=jnp.int32), jnp.array(candidate_j, dtype=jnp.int32)


def select_candidate_edges_topk(
    base_logits: jnp.ndarray,
    k: int,
    local_window: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select top-k candidate edges per token based on attention logits.

    From Section 7: "For each token i, let C(i) be a set of k << T indices
    (top-k under base logits, a local window, or a syntax neighborhood)"

    Args:
        base_logits: Attention logits, shape (S, S)
        k: Number of candidates per token
        local_window: Also include local window edges

    Returns:
        candidate_i: Source indices
        candidate_j: Target indices
    """
    S = base_logits.shape[0]

    # Mask self-attention
    logits_masked = base_logits - 1e9 * jnp.eye(S)

    # Get top-k per row
    top_k_indices = jnp.argsort(-logits_masked, axis=1)[:, :k]

    candidate_i = []
    candidate_j = []

    for i in range(S):
        # Add top-k
        for j in top_k_indices[i]:
            candidate_i.append(i)
            candidate_j.append(int(j))

        # Add local window
        for j in range(max(0, i - local_window), min(S, i + local_window + 1)):
            if i != j and j not in top_k_indices[i]:
                candidate_i.append(i)
                candidate_j.append(j)

    return jnp.array(candidate_i, dtype=jnp.int32), jnp.array(candidate_j, dtype=jnp.int32)


# ============================================================
# Template Functions
# ============================================================


def compute_soft_templates(
    Xw: jnp.ndarray, Zc: jnp.ndarray, tau: float, gamma_ctx: float = 0.0
) -> jnp.ndarray:
    """Compute soft template assignments P(c|w) via softmax."""
    scores = Xw @ Zc.T
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
    """Derive soft node states p_i from template assignments."""
    template_norms = jnp.linalg.norm(Zc, axis=1)
    weighted_norm = jnp.sum(psi * template_norms[None, :], axis=1)
    p = jax.nn.sigmoid(weighted_norm / temperature)
    return p


def predict_relation_weights_dense(
    psi: jnp.ndarray,
    Zc: jnp.ndarray,
    W_rel: jnp.ndarray,
) -> jnp.ndarray:
    """Predict relation-type mixture weights for all token pairs."""
    emb = psi @ Zc  # (S, D)
    S, D = emb.shape

    emb_i = jnp.broadcast_to(emb[:, None, :], (S, S, D))
    emb_j = jnp.broadcast_to(emb[None, :, :], (S, S, D))
    pair_features = jnp.concatenate([emb_i, emb_j], axis=-1)

    logits = pair_features @ W_rel
    weights = jax.nn.softmax(logits, axis=-1)

    return weights


# ============================================================
# Hamiltonian Attention Bias with Edge Masks
# ============================================================


def compute_hamiltonian_attention_bias_masked(
    p: jnp.ndarray,
    m: jnp.ndarray,
    w_rel: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Compute Hamiltonian attention bias with soft edge masks (Eq. 13).

    The edge mask m_ij gates whether the relational bias applies to each pair.

        bias_ij = -gamma * m_ij * Σ_r w_ij,r * E_r(p_i, p_j)

    Args:
        p: Soft node states, shape (S,)
        m: Soft edge masks, shape (S, S)
        w_rel: Relation mixture weights, shape (S, S, n_relations)
        H_r: Hamiltonian matrices, shape (n_relations, 2, 2)
        k_r: Relation offsets, shape (n_relations,)
        gamma: Bias strength

    Returns:
        Attention bias matrix, shape (S, S)
    """
    E_all = relaxed_pair_energy_batched(p, H_r, k_r)
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)

    # Apply edge masks: only add bias where mask is high
    bias = -gamma * m * E_weighted

    return bias


# ============================================================
# Multi-Head Attention with Full Phase 2 Support
# ============================================================


def multihead_attention_hamiltonian_v2(
    Eseg: jnp.ndarray,
    psi: jnp.ndarray,
    p: jnp.ndarray,
    m: jnp.ndarray,
    w_rel: jnp.ndarray,
    W_psi: jnp.ndarray,
    J_pmi: jnp.ndarray,
    idf_seg: jnp.ndarray,
    params: ModelParams,
    head_dim: int,
    local_window: int,
    gamma_hamiltonian: float,
    alpha_template: float,
    beta_directional: float,
    lambda_idf: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Multi-head attention with soft edge masks and Hamiltonian biases.

    Phase 2 addition: Edge masks m_ij gate which pairs receive relational bias.
    """
    S, D = Eseg.shape
    n_heads = params.Wq.shape[0]

    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    local_mask = dist <= local_window

    # Legacy biases
    tpl_bias = (psi @ W_psi) @ psi.T
    dir_bias = (psi @ J_pmi) @ psi.T
    idf_tgt = idf_seg[None, :]

    # Phase 2: Hamiltonian bias with edge masks
    hamiltonian_bias = compute_hamiltonian_attention_bias_masked(
        p, m, w_rel, params.H_r, params.k_r, gamma=gamma_hamiltonian
    )

    def single_head_attention(head_idx):
        Wq_h = params.Wq[head_idx]
        Wk_h = params.Wk[head_idx]
        Wv_h = params.Wv[head_idx]

        Q = Eseg @ Wq_h
        K = Eseg @ Wk_h
        V = Eseg @ Wv_h

        base_logits = (Q @ K.T) / jnp.sqrt(head_dim)

        logits = (
            base_logits
            + hamiltonian_bias
            + alpha_template * tpl_bias
            + beta_directional * dir_bias
            + lambda_idf * idf_tgt
        )

        logits = jnp.where(local_mask, logits, -1e9)
        logits = logits - 1e9 * jnp.eye(S, dtype=logits.dtype)

        A = jax.nn.softmax(logits, axis=1)
        out = A @ V

        return out, A, base_logits

    head_outputs = []
    head_attns = []
    head_base_logits = []

    for h in range(n_heads):
        out_h, A_h, base_h = single_head_attention(h)
        head_outputs.append(out_h)
        head_attns.append(A_h)
        head_base_logits.append(base_h)

    concat_out = jnp.concatenate(head_outputs, axis=1)
    output = concat_out @ params.Wo
    A_combined = jnp.stack(head_attns, axis=0).mean(axis=0)

    E_all = relaxed_pair_energy_batched(p, params.H_r, params.k_r)
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)

    diagnostics = {
        "head_attns": jnp.stack(head_attns, axis=0),
        "head_base_logits": jnp.stack(head_base_logits, axis=0),
        "tpl_bias": tpl_bias,
        "dir_bias": dir_bias,
        "hamiltonian_bias": hamiltonian_bias,
        "edge_masks": m,
        "E_weighted": E_weighted,
        "E_per_relation": E_all,
        "w_rel": w_rel,
    }

    return output, A_combined, diagnostics


# ============================================================
# Loss Functions with F_rel
# ============================================================


def template_entropy(P: jnp.ndarray) -> jnp.ndarray:
    """Compute mean entropy of template assignments."""
    return -jnp.mean(jnp.sum(P * jnp.log(P + 1e-12), axis=1))


def combined_loss_with_frel(
    Zc: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
    W_mask_1: jnp.ndarray,
    b_mask_1: jnp.ndarray,
    W_mask_2: jnp.ndarray,
    b_mask_2: jnp.ndarray,
    W_rel: jnp.ndarray,
    Xw: jnp.ndarray,
    edges_i: jnp.ndarray,
    edges_j: jnp.ndarray,
    edges_w: jnp.ndarray,
    tau: float,
    entropy_weight: float,
    binarization_weight: float,
    frel_weight: float,
) -> jnp.ndarray:
    """
    Combined loss including F_rel (Eq. 12 simplified).

    F_total = F_clustering + λ_rel * F_rel + F_bin

    For template learning, we use co-occurrence edges as the candidate set.
    """
    P = compute_soft_templates(Xw, Zc, tau)

    # Edge agreement loss
    Pi = P[edges_i]
    Pj = P[edges_j]
    agreement = jnp.sum(Pi * Pj, axis=1)
    edge_loss = -jnp.sum(edges_w * agreement)

    # Entropy regularization
    ent = template_entropy(P)
    ent_loss = -entropy_weight * ent

    # Binarization regularizer
    bin_loss = binarization_regularizer(P, beta=binarization_weight)

    # F_rel on vocabulary edges (subset for efficiency)
    # Use template-weighted embeddings
    emb = P @ Zc  # (V, D)

    # Compute soft node states
    template_norms = jnp.linalg.norm(Zc, axis=1)
    weighted_norm = jnp.sum(P * template_norms[None, :], axis=1)
    p = jax.nn.sigmoid(weighted_norm)

    # Sample edges for F_rel computation (use first N edges)
    max_frel_edges = 10000
    n_edges = min(len(edges_i), max_frel_edges)

    if n_edges > 0:
        # Get embeddings for sampled edges
        emb_i = emb[edges_i[:n_edges]]
        emb_j = emb[edges_j[:n_edges]]

        # Compute edge features (using indices as proxy for position)
        edge_features = jnp.zeros((n_edges, 4))  # Simplified features

        # Predict edge masks
        x = jnp.concatenate([emb_i, emb_j, edge_features], axis=-1)
        h = jnp.tanh(x @ W_mask_1 + b_mask_1)
        mask_logits = (h @ W_mask_2 + b_mask_2).squeeze(-1)
        m = jax.nn.sigmoid(mask_logits)

        # Predict relation weights
        pair_features = jnp.concatenate([emb_i, emb_j], axis=-1)
        rel_logits = pair_features @ W_rel
        w_rel = jax.nn.softmax(rel_logits, axis=-1)

        # Compute pair energies
        p_i = p[edges_i[:n_edges]]
        p_j = p[edges_j[:n_edges]]

        h11 = H_r[:, 0, 0]
        h12 = H_r[:, 0, 1]
        h21 = H_r[:, 1, 0]
        h22 = H_r[:, 1, 1]

        E_per_rel = (
            h11[None, :] * p_i[:, None] +
            (h12 + h21)[None, :] * p_i[:, None] * p_j[:, None] +
            h22[None, :] * p_j[:, None] +
            k_r[None, :]
        )

        E_weighted = jnp.sum(w_rel * E_per_rel, axis=-1)
        F_rel = jnp.sum(m * E_weighted) / n_edges
    else:
        F_rel = 0.0

    return edge_loss + ent_loss + bin_loss + frel_weight * F_rel


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Causal Attention with Hamiltonian Constraints - Phase 2: Soft Edge Masks"
    )

    # Data
    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--cooc_window", type=int, default=6)

    # Templates
    ap.add_argument("--n_clusters", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--template_lr", type=float, default=0.01)
    ap.add_argument("--template_steps", type=int, default=100)
    ap.add_argument("--entropy_weight", type=float, default=0.1)
    ap.add_argument("--binarization_weight", type=float, default=0.05)

    # Hamiltonian relations
    ap.add_argument("--n_relations", type=int, default=6)
    ap.add_argument("--gamma_hamiltonian", type=float, default=0.5)

    # Phase 2: Edge masks and F_rel
    ap.add_argument("--mask_hidden_dim", type=int, default=32,
                    help="Hidden dimension for edge mask MLP")
    ap.add_argument("--mask_temperature", type=float, default=1.0,
                    help="Temperature for edge mask sigmoid")
    ap.add_argument("--frel_weight", type=float, default=0.1,
                    help="Weight for F_rel in combined loss")

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

    # Legacy bias weights
    ap.add_argument("--alpha_template", type=float, default=0.5)
    ap.add_argument("--beta_directional", type=float, default=0.4)
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
    print("Phase 2: Soft Edge Masks and F_rel")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"\nPhase 2 parameters:")
    print(f"  Mask hidden dim: {args.mask_hidden_dim}")
    print(f"  Mask temperature: {args.mask_temperature}")
    print(f"  F_rel weight: {args.frel_weight}")
    print(f"\nHamiltonian parameters:")
    print(f"  Relations: {args.n_relations} ({', '.join(RELATION_NAMES[:args.n_relations])})")
    print(f"  Gamma (bias strength): {args.gamma_hamiltonian}")

    # Load text and build vocabulary
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

    # Build embeddings and edges
    print("\nBuilding word embeddings...")
    Xw = build_word_embeddings(
        vocab, args.emb_dim, args.ngram_min, args.ngram_max,
        args.hash_buckets, args.ngram_scale, args.seed,
    )

    print("Building co-occurrence edges...")
    edges_i, edges_j, edges_w = build_cooccurrence_edges(tok_ids, V, args.cooc_window)
    print(f"Directional edges: {len(edges_i)}")

    # Initialize parameters
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    params = init_params(
        init_key, args.n_clusters, args.emb_dim, args.n_heads, args.head_dim,
        args.n_relations, args.mask_hidden_dim, n_edge_features=4,
    )

    print(f"\nInitialized edge mask MLP:")
    print(f"  Layer 1: ({2 * args.emb_dim + 4}, {args.mask_hidden_dim})")
    print(f"  Layer 2: ({args.mask_hidden_dim}, 1)")

    # Train with combined loss including F_rel
    print(f"\nLearning templates with F_rel ({args.template_steps} steps)...")

    # Joint optimization of templates and edge mask parameters
    trainable = {
        "Zc": params.Zc,
        "W_mask_1": params.W_mask_1,
        "b_mask_1": params.b_mask_1,
        "W_mask_2": params.W_mask_2,
        "b_mask_2": params.b_mask_2,
        "W_rel": params.W_rel,
    }

    optimizer = optax.adam(args.template_lr)
    opt_state = optimizer.init(trainable)

    def loss_fn(trainable):
        return combined_loss_with_frel(
            trainable["Zc"],
            params.H_r,
            params.k_r,
            trainable["W_mask_1"],
            trainable["b_mask_1"],
            trainable["W_mask_2"],
            trainable["b_mask_2"],
            trainable["W_rel"],
            Xw,
            edges_i,
            edges_j,
            edges_w,
            args.tau,
            args.entropy_weight,
            args.binarization_weight,
            args.frel_weight,
        )

    @jit
    def train_step(trainable, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(trainable)
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, loss

    for step in range(args.template_steps):
        trainable, opt_state, loss = train_step(trainable, opt_state)
        if (step + 1) % 20 == 0 or step == 0:
            P = compute_soft_templates(Xw, trainable["Zc"], args.tau)
            ent = float(template_entropy(P))
            bin_loss = float(binarization_regularizer(P, args.binarization_weight))
            print(f"  step {step+1}/{args.template_steps}  loss={float(loss):.4f}  "
                  f"entropy={ent:.3f}  bin_loss={bin_loss:.4f}")

    # Update params
    params = params._replace(
        Zc=trainable["Zc"],
        W_mask_1=trainable["W_mask_1"],
        b_mask_1=trainable["b_mask_1"],
        W_mask_2=trainable["W_mask_2"],
        b_mask_2=trainable["b_mask_2"],
        W_rel=trainable["W_rel"],
    )

    # Compute template matrices
    P = compute_soft_templates(Xw, params.Zc, args.tau)
    W_psi = compute_template_compatibility(params.Zc)
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

    # Sample segments and run attention
    starts = [random.randint(0, T - args.seg_len - 1) for _ in range(args.n_segs)]

    print("\n" + "-" * 80)
    print(f"PHASE 2: HAMILTONIAN ATTENTION WITH SOFT EDGE MASKS ({args.n_heads} heads)")
    print("-" * 80)

    template_flow = np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
    per_head_flow = [
        np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
        for _ in range(args.n_heads)
    ]
    all_edges = []

    energy_stats = {r: [] for r in range(args.n_relations)}
    mask_stats = []

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start : start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]

        Ew = Xw[seg]
        psi = P[seg]
        idf_seg = idf_j[seg]

        # Soft node states
        p = compute_node_soft_states(psi, params.Zc, temperature=1.0)

        # Edge features for dense computation
        edge_features = compute_edge_features_dense(args.seg_len)

        # Predict edge masks (Phase 2)
        key, noise_key = jax.random.split(key)
        eps = args.token_noise * jax.random.normal(noise_key, Ew.shape)
        Eseg = Ew + (psi @ params.Zc) + eps

        m = predict_edge_masks_dense(Eseg, edge_features, params, args.mask_temperature)
        mask_stats.append(float(jnp.mean(m)))

        # Predict relation weights
        w_rel = predict_relation_weights_dense(psi, params.Zc, params.W_rel)

        # Run attention
        output, A_combined, diagnostics = multihead_attention_hamiltonian_v2(
            Eseg, psi, p, m, w_rel, W_psi, J_pmi, idf_seg,
            params, args.head_dim, args.local_window,
            args.gamma_hamiltonian, args.alpha_template,
            args.beta_directional, args.lambda_idf,
        )

        # Collect energy statistics
        E_per_rel = np.array(diagnostics["E_per_relation"])
        for r in range(args.n_relations):
            energy_stats[r].append(float(np.mean(E_per_rel[:, :, r])))

        # Accumulate flows
        A_np = np.array(A_combined)
        psi_np = np.array(psi)
        pair_mass = psi_np.T @ A_np @ psi_np
        template_flow += pair_mass

        head_attns = np.array(diagnostics["head_attns"])
        for h in range(args.n_heads):
            per_head_flow[h] += psi_np.T @ head_attns[h] @ psi_np

        # Top edges
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

            m_np = np.array(m)
            all_edges.append({
                "seg": seg_i,
                "w_i": seg_words[i],
                "w_j": seg_words[j],
                "score": float(flat_idf[idx]),
                "attn": float(A_np[i, j]),
                "mask": float(m_np[i, j]),  # Phase 2: include edge mask
                "dist": abs(i - j),
                "idf_i": float(idf[wi]),
                "idf_j": float(idf[wj]),
            })

    template_flow /= max(1, args.n_segs)
    for h in range(args.n_heads):
        per_head_flow[h] /= max(1, args.n_segs)

    # Output results
    print("\n" + "-" * 80)
    print("EDGE MASK STATISTICS")
    print("-" * 80)
    print(f"  Mean edge mask value: {np.mean(mask_stats):.4f}")
    print(f"  Std edge mask value:  {np.std(mask_stats):.4f}")

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
        print(f"Template {a:02d} -> Template {b:02d}   "
              f"flow={val:.4f}   J_pmi={J_pmi_np[a,b]:+.3f}   W_psi={float(W_psi[a,b]):.3f}")

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
    print(f"TOP {args.top_edges} TOKEN->TOKEN LINKS (with edge masks)")
    print("-" * 80)
    all_edges.sort(key=lambda r: r["score"], reverse=True)
    for r in all_edges[: args.top_edges]:
        near = "LOCAL" if r["dist"] <= args.local_window else "LONG"
        print(
            f"[seg {r['seg']}] {near:>5}  {r['w_i']:<14} -> {r['w_j']:<14}  "
            f"score={r['score']:.4f}  attn={r['attn']:.4f}  mask={r['mask']:.3f}  "
            f"idf=({r['idf_i']:.2f},{r['idf_j']:.2f})  dist={r['dist']}"
        )

    # Template word associations
    print("\n" + "-" * 80)
    print("TEMPLATE WORD ASSOCIATIONS")
    print("-" * 80)

    P_np = np.array(P)
    for c in range(args.n_clusters):
        scores = P_np[:, c] * idf
        top_idx = np.argsort(-scores)[:12]
        words = [vocab[i] for i in top_idx if not is_punct_heavy(vocab[i])][:8]
        print(f"Template {c:02d}: {', '.join(words)}")

    print("\n" + "=" * 80)
    print("Phase 2 Implementation Complete")
    print("=" * 80)
    print("Key additions from Causal-HNet paper Section 5:")
    print("  - Soft edge masks m_ij(u) via learned MLP (Section 5.1)")
    print("  - Edge feature computation (distance, direction)")
    print("  - Full relational energy F_rel (Eq. 11)")
    print("  - Combined loss with F_rel term")
    print("  - Masked Hamiltonian attention bias")
    print("\nNext phase to implement:")
    print("  - Phase 3: Predictive-coding integration (Section 2.1)")
    print("  - Phase 4: Context-gated modules (Section 2.2)")


if __name__ == "__main__":
    main()
