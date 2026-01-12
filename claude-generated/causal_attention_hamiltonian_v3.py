"""
Causal Multi-Head Attention with Hamiltonian Relational Constraints

Phase 3 Implementation based on Causal-HNet paper (Goertzel, Dec 2025)

Building on Phases 1-2, this adds:
1. Multi-layer latent states z = {z^(ℓ)}_{ℓ=1}^L (Section 2.1)
2. Predictive-coding free energy F_PC (Eq. 1)
3. Latent inference via gradient descent (Eq. 2)
4. Combined objective F_total = F_PC + λF_rel + F_bin (Eq. 12)
5. Relational module driven by chosen layer ℓ* (Section 6)

Key insight from Section 2.1: "Inference updates latents by gradient descent...
Learning updates parameters by gradient descent (possibly after inference converges)"

This creates an inner loop (latent inference) and outer loop (parameter learning).
"""

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, NamedTuple, Optional
from functools import partial

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
    h = 2166136261
    for ch in x:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# ============================================================
# Hamiltonian Relation Types
# ============================================================

RELATION_NAMES = [
    "IMPLIES", "EXCLUDES", "REQUIRES", "COOCCURS", "PRECEDES", "FOLLOWS",
]


def get_default_hamiltonian_matrices(n_relations: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize H_r matrices with semantically meaningful defaults."""
    H_r = np.zeros((n_relations, 2, 2), dtype=np.float32)
    k_r = np.zeros(n_relations, dtype=np.float32)

    if n_relations > 0:
        H_r[0] = [[1, -0.5], [-0.5, 0]]  # IMPLIES
    if n_relations > 1:
        H_r[1] = [[0, 0.5], [0.5, 0]]   # EXCLUDES
    if n_relations > 2:
        H_r[2] = [[0, -0.5], [-0.5, 1]] # REQUIRES
    if n_relations > 3:
        H_r[3] = [[1, -1], [-1, 1]]     # COOCCURS
    if n_relations > 4:
        H_r[4] = [[0.2, -0.3], [-0.7, 0.2]]  # PRECEDES
    if n_relations > 5:
        H_r[5] = [[0.2, -0.7], [-0.3, 0.2]]  # FOLLOWS

    return H_r, k_r


# ============================================================
# Model Parameters with Predictive-Coding Extensions
# ============================================================


class ModelParams(NamedTuple):
    """
    Learnable parameters for the causal attention model.

    Phase 3 adds predictive-coding layer parameters.
    """
    # Template parameters
    Zc: jnp.ndarray      # Template prototypes: (n_clusters, emb_dim)

    # Attention parameters
    Wq: jnp.ndarray      # Query projections: (n_heads, emb_dim, head_dim)
    Wk: jnp.ndarray      # Key projections: (n_heads, emb_dim, head_dim)
    Wv: jnp.ndarray      # Value projections: (n_heads, emb_dim, head_dim)
    Wo: jnp.ndarray      # Output projection: (n_heads * head_dim, emb_dim)

    # Phase 1: Hamiltonian relation parameters
    H_r: jnp.ndarray     # Relation matrices: (n_relations, 2, 2)
    k_r: jnp.ndarray     # Relation offsets: (n_relations,)
    W_rel: jnp.ndarray   # Relation mixture predictor: (2 * emb_dim, n_relations)

    # Phase 2: Soft edge mask parameters
    W_mask_1: jnp.ndarray
    b_mask_1: jnp.ndarray
    W_mask_2: jnp.ndarray
    b_mask_2: jnp.ndarray

    # Phase 3: Predictive-coding layer parameters (Section 2.1)
    # Each layer ℓ has: f_ℓ(z^(ℓ)) predicts z^(ℓ-1)
    W_pc: jnp.ndarray    # Layer weights: (n_layers, latent_dim, latent_dim)
    b_pc: jnp.ndarray    # Layer biases: (n_layers, latent_dim)
    sigma_pc: jnp.ndarray  # Layer noise std: (n_layers,)

    # Projection from input to latent space
    W_input: jnp.ndarray   # (emb_dim, latent_dim)
    b_input: jnp.ndarray   # (latent_dim,)

    # Projection from latent to output/relational space
    W_output: jnp.ndarray  # (latent_dim, emb_dim)
    b_output: jnp.ndarray  # (emb_dim,)


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
    n_pc_layers: int = 3,
    latent_dim: int = 64,
) -> ModelParams:
    """Initialize model parameters including predictive-coding layers."""
    keys = jax.random.split(key, 20)

    # Template and attention parameters
    Zc = jax.random.normal(keys[0], (n_clusters, emb_dim)) * 0.5
    Wq = jax.random.normal(keys[1], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wk = jax.random.normal(keys[2], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wv = jax.random.normal(keys[3], (n_heads, emb_dim, head_dim)) / math.sqrt(emb_dim)
    Wo = jax.random.normal(keys[4], (n_heads * head_dim, emb_dim)) / math.sqrt(n_heads * head_dim)

    # Hamiltonian matrices
    H_r_np, k_r_np = get_default_hamiltonian_matrices(n_relations)
    H_r = jnp.array(H_r_np)
    k_r = jnp.array(k_r_np)
    W_rel = jax.random.normal(keys[5], (2 * emb_dim, n_relations)) / math.sqrt(2 * emb_dim)

    # Edge mask MLP
    mask_input_dim = 2 * emb_dim + n_edge_features
    W_mask_1 = jax.random.normal(keys[6], (mask_input_dim, mask_hidden_dim)) / math.sqrt(mask_input_dim)
    b_mask_1 = jnp.zeros(mask_hidden_dim)
    W_mask_2 = jax.random.normal(keys[7], (mask_hidden_dim, 1)) / math.sqrt(mask_hidden_dim)
    b_mask_2 = jnp.zeros(1)

    # Phase 3: Predictive-coding layers
    # Each layer predicts the layer below: f_ℓ(z^(ℓ)) → z^(ℓ-1)
    W_pc = jax.random.normal(keys[8], (n_pc_layers, latent_dim, latent_dim)) / math.sqrt(latent_dim)
    b_pc = jnp.zeros((n_pc_layers, latent_dim))

    # Initialize sigma (noise std) to reasonable values
    sigma_pc = jnp.ones(n_pc_layers) * 1.0

    # Input projection: embed_dim -> latent_dim
    W_input = jax.random.normal(keys[9], (emb_dim, latent_dim)) / math.sqrt(emb_dim)
    b_input = jnp.zeros(latent_dim)

    # Output projection: latent_dim -> emb_dim
    W_output = jax.random.normal(keys[10], (latent_dim, emb_dim)) / math.sqrt(latent_dim)
    b_output = jnp.zeros(emb_dim)

    return ModelParams(
        Zc=Zc, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        H_r=H_r, k_r=k_r, W_rel=W_rel,
        W_mask_1=W_mask_1, b_mask_1=b_mask_1,
        W_mask_2=W_mask_2, b_mask_2=b_mask_2,
        W_pc=W_pc, b_pc=b_pc, sigma_pc=sigma_pc,
        W_input=W_input, b_input=b_input,
        W_output=W_output, b_output=b_output,
    )


# ============================================================
# Phase 3: Predictive-Coding Free Energy (Section 2.1)
# ============================================================


def pc_layer_forward(z_above: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Layer prediction function f_ℓ(z^(ℓ)).

    Predicts the layer below from the layer above.
    Uses tanh nonlinearity for bounded predictions.

    Args:
        z_above: Latent state at layer ℓ, shape (S, latent_dim)
        W: Weight matrix, shape (latent_dim, latent_dim)
        b: Bias vector, shape (latent_dim,)

    Returns:
        Prediction of z^(ℓ-1), shape (S, latent_dim)
    """
    return jnp.tanh(z_above @ W + b)


def compute_pc_prediction_error(
    z_below: jnp.ndarray,
    z_above: jnp.ndarray,
    W: jnp.ndarray,
    b: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """
    Compute prediction error energy for one layer.

    From Eq. 1: (1/2σ²) ||z^(ℓ-1) - f_ℓ(z^(ℓ))||²

    Args:
        z_below: Target (z^(ℓ-1)), shape (S, latent_dim)
        z_above: Source (z^(ℓ)), shape (S, latent_dim)
        W, b: Layer parameters
        sigma: Noise standard deviation

    Returns:
        Scalar prediction error energy
    """
    prediction = pc_layer_forward(z_above, W, b)
    error = z_below - prediction
    return 0.5 / (sigma ** 2 + 1e-6) * jnp.sum(error ** 2)


def compute_F_PC(
    x: jnp.ndarray,
    z_layers: List[jnp.ndarray],
    params: ModelParams,
) -> jnp.ndarray:
    """
    Compute predictive-coding free energy F_PC from Equation 1.

        F_PC(x, z; θ) = Σ_{ℓ=1}^L (1/2σ²_ℓ) ||z^(ℓ-1) - f_ℓ(z^(ℓ); θ_ℓ)||²

    where z^(0) = x (the input data projected to latent space).

    Args:
        x: Input data projected to latent space, shape (S, latent_dim)
        z_layers: List of latent states [z^(1), z^(2), ..., z^(L)]
        params: Model parameters

    Returns:
        Scalar free energy
    """
    n_layers = len(z_layers)
    F_PC = 0.0

    # z^(0) = x (input)
    z_below = x

    for ell in range(n_layers):
        z_above = z_layers[ell]
        W = params.W_pc[ell]
        b = params.b_pc[ell]
        sigma = params.sigma_pc[ell]

        F_PC += compute_pc_prediction_error(z_below, z_above, W, b, sigma)
        z_below = z_above

    return F_PC


def compute_F_PC_with_priors(
    x: jnp.ndarray,
    z_layers: List[jnp.ndarray],
    params: ModelParams,
    prior_weight: float = 0.01,
) -> jnp.ndarray:
    """
    F_PC with additional prior regularization on latents.

    Adds a simple Gaussian prior: prior(z) = (prior_weight/2) Σ ||z^(ℓ)||²
    """
    F_PC = compute_F_PC(x, z_layers, params)

    # Prior: encourage small latent magnitudes
    prior_loss = 0.0
    for z in z_layers:
        prior_loss += prior_weight * 0.5 * jnp.sum(z ** 2)

    return F_PC + prior_loss


def init_latent_states(
    x: jnp.ndarray,
    n_layers: int,
    params: ModelParams,
    key: jax.random.PRNGKey,
    noise_scale: float = 0.1,
) -> List[jnp.ndarray]:
    """
    Initialize latent states for inference.

    Strategy: Initialize each layer as a noisy copy of the input,
    allowing the inference process to refine them.

    Args:
        x: Input in latent space, shape (S, latent_dim)
        n_layers: Number of latent layers
        params: Model parameters
        key: Random key for noise
        noise_scale: Scale of initialization noise

    Returns:
        List of latent states [z^(1), ..., z^(L)]
    """
    z_layers = []
    keys = jax.random.split(key, n_layers)

    for ell in range(n_layers):
        # Initialize with forward pass + noise
        if ell == 0:
            z_init = x
        else:
            z_init = z_layers[ell - 1]

        # Add noise
        noise = noise_scale * jax.random.normal(keys[ell], z_init.shape)
        z_layers.append(z_init + noise)

    return z_layers


def latent_inference_step(
    x: jnp.ndarray,
    z_layers: List[jnp.ndarray],
    params: ModelParams,
    lr_z: float,
    prior_weight: float = 0.01,
) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
    """
    Single step of latent inference via gradient descent (Eq. 2).

        z ← z - η_z ∇_z F_PC(x, z; θ)

    Args:
        x: Input in latent space
        z_layers: Current latent states
        params: Model parameters (fixed during inference)
        lr_z: Learning rate for latent updates
        prior_weight: Weight for prior regularization

    Returns:
        Updated z_layers, current F_PC value
    """
    # Stack z_layers into a single array for gradient computation
    z_stacked = jnp.stack(z_layers, axis=0)

    def F_PC_fn(z_flat):
        # Unstack
        z_list = [z_flat[i] for i in range(z_flat.shape[0])]
        return compute_F_PC_with_priors(x, z_list, params, prior_weight)

    F_val, z_grad = jax.value_and_grad(F_PC_fn)(z_stacked)

    # Gradient descent on latents
    z_stacked_new = z_stacked - lr_z * z_grad

    # Unstack back to list
    z_layers_new = [z_stacked_new[i] for i in range(z_stacked_new.shape[0])]

    return z_layers_new, F_val


def run_latent_inference(
    x: jnp.ndarray,
    z_layers_init: List[jnp.ndarray],
    params: ModelParams,
    n_inference_steps: int,
    lr_z: float,
    prior_weight: float = 0.01,
    convergence_tol: float = 1e-4,
) -> Tuple[List[jnp.ndarray], List[float]]:
    """
    Run iterative latent inference until convergence or max steps.

    This is the "inner loop" of predictive coding - finding the latent
    states that minimize prediction error given fixed parameters.

    Args:
        x: Input in latent space
        z_layers_init: Initial latent states
        params: Model parameters (fixed)
        n_inference_steps: Maximum inference iterations
        lr_z: Learning rate for latents
        prior_weight: Prior regularization weight
        convergence_tol: Stop if F_PC change < this

    Returns:
        Converged z_layers, history of F_PC values
    """
    z_layers = z_layers_init
    F_history = []
    prev_F = float('inf')

    for step in range(n_inference_steps):
        z_layers, F_val = latent_inference_step(
            x, z_layers, params, lr_z, prior_weight
        )
        F_history.append(float(F_val))

        # Check convergence
        if abs(prev_F - float(F_val)) < convergence_tol:
            break
        prev_F = float(F_val)

    return z_layers, F_history


# ============================================================
# Combined Free Energy F_total (Section 6, Eq. 12)
# ============================================================


def relaxed_pair_energy_batched(
    p: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
) -> jnp.ndarray:
    """Compute pairwise Hamiltonian energies for all pairs."""
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
    """Binarization regularizer (Eq. 10)."""
    return beta * jnp.sum(p * (1 - p))


def compute_soft_node_states_from_latent(
    z_star: jnp.ndarray,
    W_output: jnp.ndarray,
    b_output: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Derive soft node states p from the chosen latent layer.

    From Section 4.1: p = σ(u/T) where u are logits from latents.

    Args:
        z_star: Latent state at layer ℓ*, shape (S, latent_dim)
        W_output, b_output: Projection to scalar logits
        temperature: Softmax temperature

    Returns:
        Soft node states p, shape (S,) in [0, 1]
    """
    # Project latent to embedding space
    u = z_star @ W_output + b_output  # (S, emb_dim)

    # Compute scalar "activation" as norm
    logits = jnp.linalg.norm(u, axis=-1)  # (S,)

    # Sigmoid with temperature
    p = jax.nn.sigmoid(logits / temperature)

    return p


def compute_edge_features_dense(seq_len: int) -> jnp.ndarray:
    """Compute edge features for all pairs."""
    idx = jnp.arange(seq_len)
    idx_i = idx[:, None]
    idx_j = idx[None, :]

    dist = idx_j - idx_i
    abs_dist = jnp.abs(dist)

    norm_dist = abs_dist / max(seq_len, 1)
    direction = jnp.sign(dist).astype(jnp.float32)
    log_dist = jnp.log1p(abs_dist) / jnp.log1p(seq_len)
    rel_pos_i = idx_i / max(seq_len - 1, 1) * jnp.ones((seq_len, seq_len))

    return jnp.stack([norm_dist, direction, log_dist, rel_pos_i], axis=-1)


def predict_edge_masks_dense(
    Eseg: jnp.ndarray,
    edge_features: jnp.ndarray,
    params: ModelParams,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Predict soft edge masks for all pairs."""
    S, D = Eseg.shape
    emb_i = jnp.broadcast_to(Eseg[:, None, :], (S, S, D))
    emb_j = jnp.broadcast_to(Eseg[None, :, :], (S, S, D))

    x = jnp.concatenate([emb_i, emb_j, edge_features], axis=-1)
    h = jnp.tanh(x @ params.W_mask_1 + params.b_mask_1)
    logits = (h @ params.W_mask_2 + params.b_mask_2).squeeze(-1)

    return jax.nn.sigmoid(logits / temperature)


def predict_relation_weights_dense(
    Eseg: jnp.ndarray,
    W_rel: jnp.ndarray,
) -> jnp.ndarray:
    """Predict relation weights for all pairs."""
    S, D = Eseg.shape
    emb_i = jnp.broadcast_to(Eseg[:, None, :], (S, S, D))
    emb_j = jnp.broadcast_to(Eseg[None, :, :], (S, S, D))
    pair_features = jnp.concatenate([emb_i, emb_j], axis=-1)

    logits = pair_features @ W_rel
    return jax.nn.softmax(logits, axis=-1)


def compute_F_rel(
    p: jnp.ndarray,
    m: jnp.ndarray,
    w_rel: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute relational energy F_rel (Eq. 11).

        F_rel = Σ_{(i,j)} m_ij Σ_r w_ij,r E_r(p_i, p_j)
    """
    E_all = relaxed_pair_energy_batched(p, H_r, k_r)
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)
    return jnp.sum(m * E_weighted)


def compute_F_total(
    x: jnp.ndarray,
    z_layers: List[jnp.ndarray],
    params: ModelParams,
    ell_star: int,
    lambda_rel: float,
    beta_bin: float,
    prior_weight: float = 0.01,
    mask_temperature: float = 1.0,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute combined free energy F_total from Equation 12.

        F_total = F_PC + λ * F_rel + F_bin

    Args:
        x: Input in latent space, shape (S, latent_dim)
        z_layers: Latent states [z^(1), ..., z^(L)]
        params: Model parameters
        ell_star: Which layer drives the relational module (0-indexed)
        lambda_rel: Weight for F_rel
        beta_bin: Weight for binarization regularizer
        prior_weight: Weight for latent priors
        mask_temperature: Temperature for edge mask sigmoid

    Returns:
        F_total: Scalar total free energy
        diagnostics: Dict with component values
    """
    S = x.shape[0]

    # F_PC: Predictive-coding free energy
    F_PC = compute_F_PC_with_priors(x, z_layers, params, prior_weight)

    # Get embeddings from chosen layer ℓ*
    z_star = z_layers[min(ell_star, len(z_layers) - 1)]
    Eseg = z_star @ params.W_output + params.b_output  # (S, emb_dim)

    # Compute soft node states p from ℓ*
    p = compute_soft_node_states_from_latent(
        z_star, params.W_output, params.b_output, temperature=1.0
    )

    # Compute edge features and masks
    edge_features = compute_edge_features_dense(S)
    m = predict_edge_masks_dense(Eseg, edge_features, params, mask_temperature)

    # Compute relation weights
    w_rel = predict_relation_weights_dense(Eseg, params.W_rel)

    # F_rel: Relational energy
    F_rel = compute_F_rel(p, m, w_rel, params.H_r, params.k_r)

    # F_bin: Binarization regularizer
    F_bin = binarization_regularizer(p, beta=beta_bin)

    # Combined
    F_total = F_PC + lambda_rel * F_rel + F_bin

    diagnostics = {
        "F_PC": F_PC,
        "F_rel": F_rel,
        "F_bin": F_bin,
        "F_total": F_total,
        "p": p,
        "m": m,
        "Eseg": Eseg,
    }

    return F_total, diagnostics


# ============================================================
# Predictive-Coding Attention Module
# ============================================================


def compute_soft_templates(
    Xw: jnp.ndarray, Zc: jnp.ndarray, tau: float
) -> jnp.ndarray:
    """Compute soft template assignments."""
    scores = Xw @ Zc.T
    return jax.nn.softmax(scores / tau, axis=1)


def compute_template_compatibility(Zc: jnp.ndarray) -> jnp.ndarray:
    """Compute template compatibility matrix."""
    Zc_norm = Zc / (jnp.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
    return Zc_norm @ Zc_norm.T


def compute_directional_coupling(
    P: jnp.ndarray,
    edges_i: jnp.ndarray,
    edges_j: jnp.ndarray,
    edges_w: jnp.ndarray,
) -> jnp.ndarray:
    """Compute J_pmi matrix."""
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


def compute_hamiltonian_attention_bias_masked(
    p: jnp.ndarray,
    m: jnp.ndarray,
    w_rel: jnp.ndarray,
    H_r: jnp.ndarray,
    k_r: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """Hamiltonian attention bias with edge masks."""
    E_all = relaxed_pair_energy_batched(p, H_r, k_r)
    E_weighted = jnp.sum(w_rel * E_all, axis=-1)
    return -gamma * m * E_weighted


def multihead_attention_pc(
    Eseg: jnp.ndarray,
    p: jnp.ndarray,
    m: jnp.ndarray,
    w_rel: jnp.ndarray,
    psi: jnp.ndarray,
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
    """Multi-head attention with predictive-coding derived features."""
    S, D = Eseg.shape
    n_heads = params.Wq.shape[0]

    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    local_mask = dist <= local_window

    tpl_bias = (psi @ W_psi) @ psi.T
    dir_bias = (psi @ J_pmi) @ psi.T
    idf_tgt = idf_seg[None, :]

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

    head_outputs, head_attns, head_base_logits = [], [], []
    for h in range(n_heads):
        out_h, A_h, base_h = single_head_attention(h)
        head_outputs.append(out_h)
        head_attns.append(A_h)
        head_base_logits.append(base_h)

    concat_out = jnp.concatenate(head_outputs, axis=1)
    output = concat_out @ params.Wo
    A_combined = jnp.stack(head_attns, axis=0).mean(axis=0)

    diagnostics = {
        "head_attns": jnp.stack(head_attns, axis=0),
        "hamiltonian_bias": hamiltonian_bias,
        "edge_masks": m,
    }

    return output, A_combined, diagnostics


# ============================================================
# Training with Predictive Coding
# ============================================================


def template_entropy(P: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(jnp.sum(P * jnp.log(P + 1e-12), axis=1))


def pc_training_loss(
    trainable: Dict,
    fixed_params: Dict,
    x_batch: jnp.ndarray,
    z_layers_batch: List[jnp.ndarray],
    ell_star: int,
    lambda_rel: float,
    beta_bin: float,
    prior_weight: float,
) -> jnp.ndarray:
    """
    Training loss incorporating F_total.

    This is used for parameter updates (outer loop).
    """
    # Reconstruct params
    params = ModelParams(
        Zc=trainable["Zc"],
        Wq=fixed_params["Wq"],
        Wk=fixed_params["Wk"],
        Wv=fixed_params["Wv"],
        Wo=fixed_params["Wo"],
        H_r=fixed_params["H_r"],
        k_r=fixed_params["k_r"],
        W_rel=trainable["W_rel"],
        W_mask_1=trainable["W_mask_1"],
        b_mask_1=trainable["b_mask_1"],
        W_mask_2=trainable["W_mask_2"],
        b_mask_2=trainable["b_mask_2"],
        W_pc=trainable["W_pc"],
        b_pc=trainable["b_pc"],
        sigma_pc=fixed_params["sigma_pc"],
        W_input=trainable["W_input"],
        b_input=trainable["b_input"],
        W_output=trainable["W_output"],
        b_output=trainable["b_output"],
    )

    F_total, _ = compute_F_total(
        x_batch, z_layers_batch, params,
        ell_star, lambda_rel, beta_bin, prior_weight
    )

    return F_total


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Causal Attention with Predictive Coding - Phase 3"
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

    # Hamiltonian
    ap.add_argument("--n_relations", type=int, default=6)
    ap.add_argument("--gamma_hamiltonian", type=float, default=0.5)

    # Phase 2: Edge masks
    ap.add_argument("--mask_hidden_dim", type=int, default=32)
    ap.add_argument("--mask_temperature", type=float, default=1.0)
    ap.add_argument("--frel_weight", type=float, default=0.1)

    # Phase 3: Predictive coding
    ap.add_argument("--n_pc_layers", type=int, default=3,
                    help="Number of predictive-coding layers")
    ap.add_argument("--latent_dim", type=int, default=64,
                    help="Dimension of latent states")
    ap.add_argument("--n_inference_steps", type=int, default=10,
                    help="Number of latent inference iterations")
    ap.add_argument("--lr_z", type=float, default=0.1,
                    help="Learning rate for latent inference")
    ap.add_argument("--ell_star", type=int, default=1,
                    help="Which layer drives relational module (0-indexed)")
    ap.add_argument("--prior_weight", type=float, default=0.01,
                    help="Weight for latent priors")

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

    # Legacy biases
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
    print("CAUSAL ATTENTION WITH PREDICTIVE CODING")
    print("Phase 3: Multi-Layer Latent Inference")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    print(f"\nPredictive-coding parameters:")
    print(f"  PC layers: {args.n_pc_layers}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Inference steps: {args.n_inference_steps}")
    print(f"  Latent LR: {args.lr_z}")
    print(f"  Relational layer ℓ*: {args.ell_star}")

    # Load data
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

    # Build embeddings
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
        n_pc_layers=args.n_pc_layers, latent_dim=args.latent_dim,
    )

    print(f"\nPredictive-coding architecture:")
    print(f"  Input projection: ({args.emb_dim}, {args.latent_dim})")
    print(f"  PC layers: {args.n_pc_layers} x ({args.latent_dim}, {args.latent_dim})")
    print(f"  Output projection: ({args.latent_dim}, {args.emb_dim})")

    # Compute template matrices
    P = compute_soft_templates(Xw, params.Zc, args.tau)
    W_psi = compute_template_compatibility(params.Zc)
    J_pmi = compute_directional_coupling(P, edges_i, edges_j, edges_w)
    J_pmi_np = np.array(J_pmi)

    # Sample segments and run PC inference + attention
    starts = [random.randint(0, T - args.seg_len - 1) for _ in range(args.n_segs)]

    print("\n" + "-" * 80)
    print(f"PREDICTIVE-CODING INFERENCE + ATTENTION ({args.n_heads} heads)")
    print("-" * 80)

    template_flow = np.zeros((args.n_clusters, args.n_clusters), dtype=np.float64)
    all_edges = []
    pc_stats = {"F_PC": [], "F_rel": [], "F_bin": [], "F_total": [], "inference_steps": []}

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start : start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]

        # Get embeddings
        Ew = Xw[seg]
        psi = P[seg]
        idf_seg = idf_j[seg]

        # Project input to latent space
        key, noise_key = jax.random.split(key)
        x_latent = Ew @ params.W_input + params.b_input  # (S, latent_dim)

        # Initialize latent states
        key, init_z_key = jax.random.split(key)
        z_layers_init = init_latent_states(
            x_latent, args.n_pc_layers, params, init_z_key, noise_scale=0.1
        )

        # Run latent inference (inner loop)
        z_layers, F_history = run_latent_inference(
            x_latent, z_layers_init, params,
            n_inference_steps=args.n_inference_steps,
            lr_z=args.lr_z,
            prior_weight=args.prior_weight,
        )

        pc_stats["inference_steps"].append(len(F_history))

        # Compute F_total with converged latents
        F_total, diag = compute_F_total(
            x_latent, z_layers, params,
            ell_star=args.ell_star,
            lambda_rel=args.frel_weight,
            beta_bin=args.binarization_weight,
            prior_weight=args.prior_weight,
            mask_temperature=args.mask_temperature,
        )

        pc_stats["F_PC"].append(float(diag["F_PC"]))
        pc_stats["F_rel"].append(float(diag["F_rel"]))
        pc_stats["F_bin"].append(float(diag["F_bin"]))
        pc_stats["F_total"].append(float(diag["F_total"]))

        # Get features from chosen layer
        z_star = z_layers[min(args.ell_star, len(z_layers) - 1)]
        Eseg = z_star @ params.W_output + params.b_output

        p = diag["p"]
        m = diag["m"]
        w_rel = predict_relation_weights_dense(Eseg, params.W_rel)

        # Run attention
        output, A_combined, attn_diag = multihead_attention_pc(
            Eseg, p, m, w_rel, psi, W_psi, J_pmi, idf_seg,
            params, args.head_dim, args.local_window,
            args.gamma_hamiltonian, args.alpha_template,
            args.beta_directional, args.lambda_idf,
        )

        # Accumulate flows
        A_np = np.array(A_combined)
        psi_np = np.array(psi)
        template_flow += psi_np.T @ A_np @ psi_np

        # Top edges
        flat_idf = (A_np * (idf[seg][:, None] * idf[seg][None, :])).ravel()
        top_idx = np.argpartition(flat_idf, -20)[-20:]
        top_idx = top_idx[np.argsort(-flat_idf[top_idx])]

        m_np = np.array(m)
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

            all_edges.append({
                "seg": seg_i,
                "w_i": seg_words[i],
                "w_j": seg_words[j],
                "score": float(flat_idf[idx]),
                "attn": float(A_np[i, j]),
                "mask": float(m_np[i, j]),
                "dist": abs(i - j),
                "idf_i": float(idf[wi]),
                "idf_j": float(idf[wj]),
            })

        if seg_i <= 2:
            print(f"\nSegment {seg_i} inference:")
            print(f"  Steps: {len(F_history)}, Final F_PC: {F_history[-1]:.4f}")
            print(f"  F_total: {float(F_total):.4f} = F_PC({float(diag['F_PC']):.4f}) "
                  f"+ λ*F_rel({float(diag['F_rel']):.4f}) + F_bin({float(diag['F_bin']):.4f})")

    template_flow /= max(1, args.n_segs)

    # Output results
    print("\n" + "-" * 80)
    print("PREDICTIVE-CODING STATISTICS")
    print("-" * 80)
    print(f"  Mean F_PC:   {np.mean(pc_stats['F_PC']):.4f} ± {np.std(pc_stats['F_PC']):.4f}")
    print(f"  Mean F_rel:  {np.mean(pc_stats['F_rel']):.4f} ± {np.std(pc_stats['F_rel']):.4f}")
    print(f"  Mean F_bin:  {np.mean(pc_stats['F_bin']):.4f} ± {np.std(pc_stats['F_bin']):.4f}")
    print(f"  Mean F_total: {np.mean(pc_stats['F_total']):.4f} ± {np.std(pc_stats['F_total']):.4f}")
    print(f"  Avg inference steps: {np.mean(pc_stats['inference_steps']):.1f}")

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
        print(f"Template {a:02d} -> Template {b:02d}   flow={val:.4f}   J_pmi={J_pmi_np[a,b]:+.3f}")

    print("\n" + "-" * 80)
    print(f"TOP {args.top_edges} TOKEN->TOKEN LINKS")
    print("-" * 80)
    all_edges.sort(key=lambda r: r["score"], reverse=True)
    for r in all_edges[: args.top_edges]:
        near = "LOCAL" if r["dist"] <= args.local_window else "LONG"
        print(
            f"[seg {r['seg']}] {near:>5}  {r['w_i']:<14} -> {r['w_j']:<14}  "
            f"score={r['score']:.4f}  attn={r['attn']:.4f}  mask={r['mask']:.3f}  "
            f"dist={r['dist']}"
        )

    # Template associations
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
    print("Phase 3 Implementation Complete")
    print("=" * 80)
    print("Key additions from Causal-HNet paper Section 2.1:")
    print("  - Multi-layer latent states z = {z^(ℓ)}_{ℓ=1}^L")
    print("  - Predictive-coding free energy F_PC (Eq. 1)")
    print("  - Latent inference via gradient descent (Eq. 2)")
    print("  - Combined objective F_total = F_PC + λF_rel + F_bin (Eq. 12)")
    print("  - Relational module driven by chosen layer ℓ*")
    print("\nNext phase to implement:")
    print("  - Phase 4: Context-gated modules (Section 2.2)")


if __name__ == "__main__":
    main()
