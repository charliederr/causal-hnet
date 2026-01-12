"""
Causal Multi-Head Attention with Hamiltonian Relational Constraints

Phase 5: Full Training Loop

Complete implementation with:
1. Joint optimization of all model parameters
2. Inner loop: Latent inference (Eq. 2)
3. Outer loop: Gated parameter updates (Eq. 4)
4. Learning rate scheduling with warmup
5. Training metrics and logging
6. Checkpoint saving/loading

Training procedure:
- Sample segment from corpus
- Run latent inference to convergence (inner loop)
- Compute F_total with converged latents
- Compute gradients and apply gated updates (outer loop)
- Log metrics periodically
"""

import argparse
import math
import random
import re
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, NamedTuple, Optional, Any
from functools import partial
from dataclasses import dataclass, asdict
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
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
# Hamiltonian Relations
# ============================================================

RELATION_NAMES = [
    "IMPLIES", "EXCLUDES", "REQUIRES", "COOCCURS", "PRECEDES", "FOLLOWS",
]


def get_default_hamiltonian_matrices(n_relations: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    H_r = np.zeros((n_relations, 2, 2), dtype=np.float32)
    k_r = np.zeros(n_relations, dtype=np.float32)

    if n_relations > 0:
        H_r[0] = [[1, -0.5], [-0.5, 0]]
    if n_relations > 1:
        H_r[1] = [[0, 0.5], [0.5, 0]]
    if n_relations > 2:
        H_r[2] = [[0, -0.5], [-0.5, 1]]
    if n_relations > 3:
        H_r[3] = [[1, -1], [-1, 1]]
    if n_relations > 4:
        H_r[4] = [[0.2, -0.3], [-0.7, 0.2]]
    if n_relations > 5:
        H_r[5] = [[0.2, -0.7], [-0.3, 0.2]]

    return H_r, k_r


# ============================================================
# Module Definitions
# ============================================================

MODULE_SPECS = [
    ("template", ["Zc"]),
    ("attention_qk", ["Wq", "Wk"]),
    ("attention_vo", ["Wv", "Wo"]),
    ("relational", ["H_r", "k_r", "W_rel"]),
    ("edge_mask", ["W_mask_1", "b_mask_1", "W_mask_2", "b_mask_2"]),
    ("pc_layer_0", []),  # Handled via W_pc indexing
    ("pc_layer_1", []),
    ("pc_layer_2", []),
    ("projections", ["W_input", "b_input", "W_output", "b_output"]),
]


def get_module_param_mapping() -> Dict[str, int]:
    return {
        "Zc": 0,
        "Wq": 1, "Wk": 1,
        "Wv": 2, "Wo": 2,
        "H_r": 3, "k_r": 3, "W_rel": 3,
        "W_mask_1": 4, "b_mask_1": 4, "W_mask_2": 4, "b_mask_2": 4,
        "W_pc": 5, "b_pc": 5,  # PC layers share module index for simplicity
        "W_input": 8, "b_input": 8, "W_output": 8, "b_output": 8,
        "W_ctx_enc": -1, "b_ctx_enc": -1, "W_gate": -1, "b_gate": -1,
        "sigma_pc": -1,
    }


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    seg_len: int = 256
    batch_size: int = 1  # Segments per batch

    # Training
    n_epochs: int = 10
    steps_per_epoch: int = 100
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 1e-4

    # Latent inference (inner loop)
    n_inference_steps: int = 10
    lr_z: float = 0.1
    prior_weight: float = 0.01

    # Loss weights
    lambda_rel: float = 0.1
    beta_bin: float = 0.05
    gate_budget: float = 4.0  # Target sum of gates (out of n_modules=9)
    gate_budget_weight: float = 0.5  # Weight for budget constraint
    gate_variance_weight: float = 0.5  # Encourage variance in gate values

    # Context gating
    gate_temperature: float = 0.5  # Lower temp = sharper gates
    ell_star: int = 1

    # Logging
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100

    # Model
    n_clusters: int = 16
    emb_dim: int = 96
    n_heads: int = 4
    head_dim: int = 32
    n_relations: int = 6
    mask_hidden_dim: int = 32
    n_pc_layers: int = 3
    latent_dim: int = 64
    ctx_dim: int = 32
    n_modules: int = 9


# ============================================================
# Model Parameters
# ============================================================

class ModelParams(NamedTuple):
    Zc: jnp.ndarray
    Wq: jnp.ndarray
    Wk: jnp.ndarray
    Wv: jnp.ndarray
    Wo: jnp.ndarray
    H_r: jnp.ndarray
    k_r: jnp.ndarray
    W_rel: jnp.ndarray
    W_mask_1: jnp.ndarray
    b_mask_1: jnp.ndarray
    W_mask_2: jnp.ndarray
    b_mask_2: jnp.ndarray
    W_pc: jnp.ndarray
    b_pc: jnp.ndarray
    sigma_pc: jnp.ndarray
    W_input: jnp.ndarray
    b_input: jnp.ndarray
    W_output: jnp.ndarray
    b_output: jnp.ndarray
    W_ctx_enc: jnp.ndarray
    b_ctx_enc: jnp.ndarray
    W_gate: jnp.ndarray
    b_gate: jnp.ndarray


def init_params(key: jax.random.PRNGKey, config: TrainConfig) -> ModelParams:
    """Initialize all model parameters."""
    keys = jax.random.split(key, 25)

    Zc = jax.random.normal(keys[0], (config.n_clusters, config.emb_dim)) * 0.5
    Wq = jax.random.normal(keys[1], (config.n_heads, config.emb_dim, config.head_dim)) / math.sqrt(config.emb_dim)
    Wk = jax.random.normal(keys[2], (config.n_heads, config.emb_dim, config.head_dim)) / math.sqrt(config.emb_dim)
    Wv = jax.random.normal(keys[3], (config.n_heads, config.emb_dim, config.head_dim)) / math.sqrt(config.emb_dim)
    Wo = jax.random.normal(keys[4], (config.n_heads * config.head_dim, config.emb_dim)) / math.sqrt(config.n_heads * config.head_dim)

    H_r_np, k_r_np = get_default_hamiltonian_matrices(config.n_relations)
    H_r = jnp.array(H_r_np)
    k_r = jnp.array(k_r_np)
    W_rel = jax.random.normal(keys[5], (2 * config.emb_dim, config.n_relations)) / math.sqrt(2 * config.emb_dim)

    mask_input_dim = 2 * config.emb_dim + 4
    W_mask_1 = jax.random.normal(keys[6], (mask_input_dim, config.mask_hidden_dim)) / math.sqrt(mask_input_dim)
    b_mask_1 = jnp.zeros(config.mask_hidden_dim)
    W_mask_2 = jax.random.normal(keys[7], (config.mask_hidden_dim, 1)) / math.sqrt(config.mask_hidden_dim)
    b_mask_2 = jnp.zeros(1)

    W_pc = jax.random.normal(keys[8], (config.n_pc_layers, config.latent_dim, config.latent_dim)) / math.sqrt(config.latent_dim)
    b_pc = jnp.zeros((config.n_pc_layers, config.latent_dim))
    sigma_pc = jnp.ones(config.n_pc_layers)

    W_input = jax.random.normal(keys[9], (config.emb_dim, config.latent_dim)) / math.sqrt(config.emb_dim)
    b_input = jnp.zeros(config.latent_dim)
    W_output = jax.random.normal(keys[10], (config.latent_dim, config.emb_dim)) / math.sqrt(config.latent_dim)
    b_output = jnp.zeros(config.emb_dim)

    W_ctx_enc = jax.random.normal(keys[11], (config.emb_dim, config.ctx_dim)) / math.sqrt(config.emb_dim)
    b_ctx_enc = jnp.zeros(config.ctx_dim)
    W_gate = jax.random.normal(keys[12], (config.ctx_dim, config.n_modules)) / math.sqrt(config.ctx_dim)
    # Initialize gate biases with varied values to encourage differentiation
    # Some modules start more active, some less
    b_gate = jax.random.uniform(keys[13], (config.n_modules,), minval=-1.0, maxval=1.0)

    return ModelParams(
        Zc=Zc, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
        H_r=H_r, k_r=k_r, W_rel=W_rel,
        W_mask_1=W_mask_1, b_mask_1=b_mask_1, W_mask_2=W_mask_2, b_mask_2=b_mask_2,
        W_pc=W_pc, b_pc=b_pc, sigma_pc=sigma_pc,
        W_input=W_input, b_input=b_input, W_output=W_output, b_output=b_output,
        W_ctx_enc=W_ctx_enc, b_ctx_enc=b_ctx_enc, W_gate=W_gate, b_gate=b_gate,
    )


def params_to_dict(params: ModelParams) -> Dict[str, jnp.ndarray]:
    """Convert NamedTuple to dict for optimizer."""
    return {k: getattr(params, k) for k in params._fields}


def dict_to_params(d: Dict[str, jnp.ndarray]) -> ModelParams:
    """Convert dict back to NamedTuple."""
    return ModelParams(**d)


# ============================================================
# Data Loading
# ============================================================

def build_word_embeddings(vocab, emb_dim, ngram_min, ngram_max, hash_buckets, ngram_scale, seed):
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


def build_cooccurrence_edges(tok_ids, V, window):
    T = len(tok_ids)
    cooc = defaultdict(float)

    for i in range(T):
        wi = tok_ids[i]
        if wi < 0:
            continue
        for j in range(i + 1, min(i + window, T)):
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


class DataLoader:
    """Simple data loader for text segments."""

    def __init__(self, tok_ids: np.ndarray, Xw: jnp.ndarray, seg_len: int, seed: int = 0):
        self.tok_ids = tok_ids
        self.Xw = Xw
        self.seg_len = seg_len
        self.T = len(tok_ids)
        self.rng = np.random.RandomState(seed)

    def sample_segment(self) -> Tuple[jnp.ndarray, np.ndarray]:
        """Sample a random segment."""
        start = self.rng.randint(0, self.T - self.seg_len)
        seg = self.tok_ids[start:start + self.seg_len].copy()
        seg[seg < 0] = 0
        Eseg = self.Xw[seg]
        return Eseg, seg

    def sample_batch(self, batch_size: int) -> List[Tuple[jnp.ndarray, np.ndarray]]:
        """Sample a batch of segments."""
        return [self.sample_segment() for _ in range(batch_size)]


# ============================================================
# Core Computations
# ============================================================

def pc_layer_forward(z, W, b):
    return jnp.tanh(z @ W + b)


def compute_F_PC(x, z_layers, params):
    F = 0.0
    z_below = x
    for ell in range(len(z_layers)):
        pred = pc_layer_forward(z_layers[ell], params.W_pc[ell], params.b_pc[ell])
        sigma = params.sigma_pc[ell]
        F += 0.5 / (sigma**2 + 1e-6) * jnp.sum((z_below - pred)**2)
        z_below = z_layers[ell]
    return F


def compute_F_PC_with_priors(x, z_layers, params, prior_weight):
    F = compute_F_PC(x, z_layers, params)
    for z in z_layers:
        F += prior_weight * 0.5 * jnp.sum(z**2)
    return F


def init_latent_states(x, n_layers, key, noise_scale=0.1):
    z_layers = []
    keys = jax.random.split(key, n_layers)
    for ell in range(n_layers):
        z_init = x if ell == 0 else z_layers[ell-1]
        noise = noise_scale * jax.random.normal(keys[ell], z_init.shape)
        z_layers.append(z_init + noise)
    return z_layers


def run_latent_inference(x, z_init, params, config):
    """Run latent inference (inner loop)."""
    z_stacked = jnp.stack(z_init, axis=0)

    def F_fn(z_flat):
        z_list = [z_flat[i] for i in range(z_flat.shape[0])]
        return compute_F_PC_with_priors(x, z_list, params, config.prior_weight)

    for _ in range(config.n_inference_steps):
        F_val, z_grad = value_and_grad(F_fn)(z_stacked)
        z_stacked = z_stacked - config.lr_z * z_grad

    return [z_stacked[i] for i in range(z_stacked.shape[0])], float(F_val)


def relaxed_pair_energy_batched(p, H_r, k_r):
    h11, h12, h21, h22 = H_r[:, 0, 0], H_r[:, 0, 1], H_r[:, 1, 0], H_r[:, 1, 1]
    p_i, p_j = p[:, None, None], p[None, :, None]
    return (h11[None, None, :] * p_i + (h12 + h21)[None, None, :] * p_i * p_j +
            h22[None, None, :] * p_j + k_r[None, None, :])


def compute_edge_features_dense(S):
    idx = jnp.arange(S)
    dist = jnp.abs(idx[:, None] - idx[None, :])
    return jnp.stack([
        dist / max(S, 1),
        jnp.sign(idx[None, :] - idx[:, None]).astype(jnp.float32),
        jnp.log1p(dist) / jnp.log1p(S),
        idx[:, None] / max(S - 1, 1) * jnp.ones((S, S)),
    ], axis=-1)


def predict_edge_masks(Eseg, edge_features, params):
    S, D = Eseg.shape
    emb_i = jnp.broadcast_to(Eseg[:, None, :], (S, S, D))
    emb_j = jnp.broadcast_to(Eseg[None, :, :], (S, S, D))
    x = jnp.concatenate([emb_i, emb_j, edge_features], axis=-1)
    h = jnp.tanh(x @ params.W_mask_1 + params.b_mask_1)
    return jax.nn.sigmoid((h @ params.W_mask_2 + params.b_mask_2).squeeze(-1))


def predict_relation_weights(Eseg, W_rel):
    S, D = Eseg.shape
    emb_i = jnp.broadcast_to(Eseg[:, None, :], (S, S, D))
    emb_j = jnp.broadcast_to(Eseg[None, :, :], (S, S, D))
    return jax.nn.softmax(jnp.concatenate([emb_i, emb_j], axis=-1) @ W_rel, axis=-1)


def encode_context(x, params):
    x_mean = jnp.mean(x, axis=0)
    return jnp.tanh(x_mean @ params.W_ctx_enc + params.b_ctx_enc)


def compute_gates(ctx, params, temperature):
    logits = ctx @ params.W_gate + params.b_gate
    return jax.nn.sigmoid(logits / temperature)


def compute_F_total(x, z_layers, params, config):
    """Compute combined free energy F_total."""
    S = x.shape[0]

    # F_PC
    F_PC = compute_F_PC_with_priors(x, z_layers, params, config.prior_weight)

    # Get embeddings from chosen layer
    z_star = z_layers[min(config.ell_star, len(z_layers) - 1)]
    Eseg = z_star @ params.W_output + params.b_output

    # Soft node states
    u_norm = jnp.linalg.norm(Eseg, axis=-1)
    p = jax.nn.sigmoid(u_norm)

    # Edge masks and relation weights
    edge_features = compute_edge_features_dense(S)
    m = predict_edge_masks(Eseg, edge_features, params)
    w_rel = predict_relation_weights(Eseg, params.W_rel)

    # F_rel (use softplus to ensure non-negative energies)
    E_all = relaxed_pair_energy_batched(p, params.H_r, params.k_r)
    E_all_pos = jax.nn.softplus(E_all)  # Ensure E_r >= 0 for proper constraint energy
    E_weighted = jnp.sum(w_rel * E_all_pos, axis=-1)
    F_rel = jnp.sum(m * E_weighted)

    # F_bin
    F_bin = config.beta_bin * jnp.sum(p * (1 - p))

    # Context and gates
    ctx = encode_context(Eseg, params)
    gates = compute_gates(ctx, params, config.gate_temperature)

    # Gate regularization:
    # 1. Budget constraint: gates should sum to target (encourages some on, some off)
    gate_sum = jnp.sum(gates)
    gate_budget_penalty = config.gate_budget_weight * (gate_sum - config.gate_budget) ** 2
    # 2. Variance incentive: encourage gates to be different from each other
    # Maximize variance = minimize negative variance
    gate_mean = jnp.mean(gates)
    gate_variance = jnp.var(gates)
    # We want high variance, so penalize low variance (minimize -variance)
    gate_variance_loss = -config.gate_variance_weight * gate_variance

    F_total = F_PC + config.lambda_rel * F_rel + F_bin + gate_budget_penalty + gate_variance_loss

    return F_total, {
        "F_PC": F_PC,
        "F_rel": F_rel,
        "F_bin": F_bin,
        "gates": gates,
        "gate_mean": gate_mean,
        "p_mean": jnp.mean(p),
        "m_mean": jnp.mean(m),
    }


# ============================================================
# Training Step
# ============================================================

def make_train_step(config: TrainConfig, optimizer):
    """Create JIT-compiled training step.

    Args:
        config: Training configuration
        optimizer: Optax optimizer (closed over, not passed to JIT)
    """

    module_mapping = get_module_param_mapping()

    def loss_fn(params_dict, x, z_layers):
        params = dict_to_params(params_dict)
        F_total, metrics = compute_F_total(x, z_layers, params, config)
        return F_total, metrics

    def apply_gated_grads(grads, gates):
        """Apply gating to gradients."""
        gated = {}
        for key, grad in grads.items():
            module_idx = module_mapping.get(key, -1)
            if module_idx >= 0 and module_idx < len(gates):
                gated[key] = grad * gates[module_idx]
            else:
                gated[key] = grad
        return gated

    @jit
    def train_step(params_dict, opt_state, x, key):
        # Project input to latent space
        params = dict_to_params(params_dict)
        x_latent = x @ params.W_input + params.b_input

        # Initialize latent states
        z_init = init_latent_states(x_latent, config.n_pc_layers, key)

        # Run latent inference (inner loop) - not differentiated through
        z_stacked = jnp.stack(z_init, axis=0)

        def F_PC_fn(z_flat):
            z_list = [z_flat[i] for i in range(z_flat.shape[0])]
            return compute_F_PC_with_priors(x_latent, z_list, params, config.prior_weight)

        for _ in range(config.n_inference_steps):
            _, z_grad = value_and_grad(F_PC_fn)(z_stacked)
            z_stacked = z_stacked - config.lr_z * z_grad

        z_layers = [jax.lax.stop_gradient(z_stacked[i]) for i in range(z_stacked.shape[0])]

        # Compute loss and gradients (outer loop)
        (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(
            params_dict, x_latent, z_layers
        )

        # Get gates for gated update
        gates = metrics["gates"]

        # Apply gated gradients
        gated_grads = apply_gated_grads(grads, gates)

        # Update parameters (optimizer is closed over)
        updates, new_opt_state = optimizer.update(gated_grads, opt_state, params_dict)
        new_params = optax.apply_updates(params_dict, updates)

        return new_params, new_opt_state, loss, metrics

    return train_step


# ============================================================
# Learning Rate Schedule
# ============================================================

def create_optimizer(config: TrainConfig):
    """Create optimizer with warmup and decay."""
    # Linear warmup then cosine decay
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )

    decay_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.n_epochs * config.steps_per_epoch - config.warmup_steps,
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule_fn, weight_decay=config.weight_decay),
    )

    return optimizer


# ============================================================
# Training Loop
# ============================================================

@dataclass
class TrainState:
    """Training state."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    loss_history: List[float] = None
    metrics_history: List[Dict] = None

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.metrics_history is None:
            self.metrics_history = []


def train(
    config: TrainConfig,
    data_loader: DataLoader,
    params: ModelParams,
    key: jax.random.PRNGKey,
    save_dir: Optional[str] = None,
) -> Tuple[ModelParams, TrainState]:
    """
    Full training loop.

    Args:
        config: Training configuration
        data_loader: Data loader for segments
        params: Initial model parameters
        key: Random key
        save_dir: Directory to save checkpoints

    Returns:
        Trained parameters, training state
    """
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {config.n_epochs}")
    print(f"Steps per epoch: {config.steps_per_epoch}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Inference steps: {config.n_inference_steps}")
    print(f"Segment length: {config.seg_len}")

    # Create optimizer
    optimizer = create_optimizer(config)
    params_dict = params_to_dict(params)
    opt_state = optimizer.init(params_dict)

    # Create training step (optimizer is closed over)
    train_step_fn = make_train_step(config, optimizer)

    # Training state
    state = TrainState()

    # Metrics accumulators
    epoch_losses = []
    epoch_metrics = defaultdict(list)

    total_steps = config.n_epochs * config.steps_per_epoch
    start_time = time.time()

    for epoch in range(config.n_epochs):
        state.epoch = epoch
        epoch_start = time.time()

        for step_in_epoch in range(config.steps_per_epoch):
            state.step = epoch * config.steps_per_epoch + step_in_epoch

            # Sample segment
            Eseg, _ = data_loader.sample_segment()

            # Get random key for this step
            key, step_key = jax.random.split(key)

            # Training step
            params_dict, opt_state, loss, metrics = train_step_fn(
                params_dict, opt_state, Eseg, step_key
            )

            # Record metrics
            loss_val = float(loss)
            epoch_losses.append(loss_val)
            state.loss_history.append(loss_val)

            for k, v in metrics.items():
                if k != "gates":
                    epoch_metrics[k].append(float(v))

            # Logging
            if state.step % config.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (state.step + 1) / elapsed if elapsed > 0 else 0
                gates_np = np.array(metrics["gates"])

                print(f"Step {state.step:5d}/{total_steps} | "
                      f"Loss: {loss_val:.4f} | "
                      f"F_PC: {float(metrics['F_PC']):.2f} | "
                      f"F_rel: {float(metrics['F_rel']):.1f} | "
                      f"Gates: [{gates_np.min():.2f}-{gates_np.max():.2f}] | "
                      f"{steps_per_sec:.1f} steps/s")

            # Save checkpoint
            if save_dir and state.step % config.save_every == 0 and state.step > 0:
                save_checkpoint(params_dict, state, config, save_dir)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses[-config.steps_per_epoch:])
        avg_F_PC = np.mean(epoch_metrics["F_PC"][-config.steps_per_epoch:])
        avg_F_rel = np.mean(epoch_metrics["F_rel"][-config.steps_per_epoch:])

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.n_epochs} complete in {epoch_time:.1f}s")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg F_PC: {avg_F_PC:.2f}")
        print(f"  Avg F_rel: {avg_F_rel:.1f}")
        print(f"{'='*60}\n")

        # Track best
        if avg_loss < state.best_loss:
            state.best_loss = avg_loss
            if save_dir:
                save_checkpoint(params_dict, state, config, save_dir, is_best=True)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best loss: {state.best_loss:.4f}")

    return dict_to_params(params_dict), state


def save_checkpoint(params_dict, state, config, save_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # Save params as numpy
    params_np = {k: np.array(v) for k, v in params_dict.items()}

    suffix = "_best" if is_best else f"_step{state.step}"
    np.savez(os.path.join(save_dir, f"params{suffix}.npz"), **params_np)

    # Save state
    state_dict = {
        "step": state.step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
    }
    with open(os.path.join(save_dir, f"state{suffix}.json"), "w") as f:
        json.dump(state_dict, f)

    print(f"  Saved checkpoint: {save_dir}/params{suffix}.npz")


def load_checkpoint(save_dir, suffix=""):
    """Load model checkpoint."""
    params_path = os.path.join(save_dir, f"params{suffix}.npz")
    params_np = dict(np.load(params_path))
    params_dict = {k: jnp.array(v) for k, v in params_np.items()}

    state_path = os.path.join(save_dir, f"state{suffix}.json")
    with open(state_path) as f:
        state_dict = json.load(f)

    state = TrainState(**state_dict)
    return dict_to_params(params_dict), state


# ============================================================
# Evaluation
# ============================================================

def evaluate(params: ModelParams, data_loader: DataLoader, config: TrainConfig, n_samples: int = 10):
    """Evaluate model on random samples."""
    metrics_list = []

    for _ in range(n_samples):
        Eseg, _ = data_loader.sample_segment()

        # Project to latent
        x_latent = Eseg @ params.W_input + params.b_input

        # Run inference
        key = jax.random.PRNGKey(0)
        z_init = init_latent_states(x_latent, config.n_pc_layers, key)
        z_layers, F_PC_final = run_latent_inference(x_latent, z_init, params, config)

        # Compute F_total
        _, metrics = compute_F_total(x_latent, z_layers, params, config)
        metrics_list.append({k: float(v) if k != "gates" else v for k, v in metrics.items()})

    # Aggregate
    avg_metrics = {}
    for k in metrics_list[0]:
        if k != "gates":
            vals = [m[k] for m in metrics_list]
            avg_metrics[k] = np.mean(vals)
            avg_metrics[f"{k}_std"] = np.std(vals)

    return avg_metrics


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Causal-HNet Full Training")

    # Data
    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--cooc_window", type=int, default=6)

    # Training
    ap.add_argument("--n_epochs", type=int, default=5)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # Latent inference
    ap.add_argument("--n_inference_steps", type=int, default=10)
    ap.add_argument("--lr_z", type=float, default=0.1)
    ap.add_argument("--prior_weight", type=float, default=0.01)

    # Loss weights
    ap.add_argument("--lambda_rel", type=float, default=0.1)
    ap.add_argument("--beta_bin", type=float, default=0.05)
    ap.add_argument("--gate_budget", type=float, default=4.0)
    ap.add_argument("--gate_budget_weight", type=float, default=0.5)
    ap.add_argument("--gate_variance_weight", type=float, default=0.5)

    # Model
    ap.add_argument("--n_clusters", type=int, default=16)
    ap.add_argument("--emb_dim", type=int, default=96)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--n_relations", type=int, default=6)
    ap.add_argument("--n_pc_layers", type=int, default=3)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--ctx_dim", type=int, default=32)
    ap.add_argument("--n_modules", type=int, default=9)
    ap.add_argument("--gate_temperature", type=float, default=1.0)
    ap.add_argument("--ell_star", type=int, default=1)

    # Segment
    ap.add_argument("--seg_len", type=int, default=256)

    # Logging
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--save_dir", type=str, default=None)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("CAUSAL-HNET FULL TRAINING")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    # Create config
    config = TrainConfig(
        seg_len=args.seg_len,
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        n_inference_steps=args.n_inference_steps,
        lr_z=args.lr_z,
        prior_weight=args.prior_weight,
        lambda_rel=args.lambda_rel,
        beta_bin=args.beta_bin,
        gate_budget=args.gate_budget,
        gate_budget_weight=args.gate_budget_weight,
        gate_variance_weight=args.gate_variance_weight,
        gate_temperature=args.gate_temperature,
        ell_star=args.ell_star,
        log_every=args.log_every,
        save_every=args.save_every,
        n_clusters=args.n_clusters,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        n_relations=args.n_relations,
        n_pc_layers=args.n_pc_layers,
        latent_dim=args.latent_dim,
        ctx_dim=args.ctx_dim,
        n_modules=args.n_modules,
    )

    # Load data
    print("\nLoading data...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().lower()

    raw_tokens = text.split()
    T = len(raw_tokens)
    print(f"Total tokens: {T}")

    freq = Counter(raw_tokens)
    vocab = [w for w, c in freq.items() if c >= args.min_freq]
    vocab = vocab[:args.max_vocab]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"Vocabulary size: {V}")

    tok_ids = np.full((T,), -1, dtype=np.int32)
    for i, w in enumerate(raw_tokens):
        tok_ids[i] = word_to_id.get(w, -1)

    # Build embeddings
    print("Building embeddings...")
    Xw = build_word_embeddings(vocab, args.emb_dim, 3, 5, 2**18, 0.2, args.seed)

    # Create data loader
    data_loader = DataLoader(tok_ids, Xw, args.seg_len, args.seed)

    # Initialize model
    print("Initializing model...")
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, config)

    # Count parameters
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {n_params:,}")

    # Train
    trained_params, state = train(
        config, data_loader, params, key, save_dir=args.save_dir
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    eval_metrics = evaluate(trained_params, data_loader, config, n_samples=20)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
