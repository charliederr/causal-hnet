import math
from collections import Counter, defaultdict

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================
# Configuration
# ============================================================

WINDOW = 5
MIN_FREQ = 5
MAX_VOCAB = 12000

N_CLUSTERS = 8
N_SWEEPS = 20           # Hamiltonian
PC_STEPS = 100
ETA_Z = 0.05

LATENT_DIM = 32
LAMBDA_H = 0.5

TOP_K = 12

# ============================================================
# Load text
# ============================================================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
T = len(tokens)

print("=" * 80)
print("HNET STEP 3 — PREDICTIVE CODING ON CLUSTERS (FIXED)")
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

# ============================================================
# Random word embeddings (evidence)
# ============================================================

key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (V, LATENT_DIM)) * 0.1

# ============================================================
# Directional co-occurrence (Step 2)
# ============================================================

cooc = defaultdict(float)

for i in range(T):
    wi = tokens[i]
    if wi not in word_to_id:
        continue
    ii = word_to_id[wi]
    for j in range(i + 1, min(i + WINDOW, T)):
        wj = tokens[j]
        if wj not in word_to_id:
            continue
        jj = word_to_id[wj]
        cooc[(ii, jj)] += 1.0

norm = np.zeros(V)
for (i, j), v in cooc.items():
    norm[i] += v

edges_i, edges_j, edges_w = [], [], []
for (i, j), v in cooc.items():
    w = v / math.sqrt(norm[i] * norm[j] + 1e-8)
    edges_i.append(i)
    edges_j.append(j)
    edges_w.append(w)

edges_i = jnp.array(edges_i, dtype=jnp.int32)
edges_j = jnp.array(edges_j, dtype=jnp.int32)
edges_w = jnp.array(edges_w, dtype=jnp.float32)

# ============================================================
# Step 1+2: Directional Hamiltonian clustering
# ============================================================

@jax.jit
def hamiltonian_sweep(state):
    S = jax.nn.one_hot(state, N_CLUSTERS)
    contrib = edges_w[:, None] * S[edges_j]
    field = jnp.zeros((V, N_CLUSTERS)).at[edges_i].add(contrib)
    return jnp.argmax(field, axis=1)

state = jax.random.randint(key, (V,), 0, N_CLUSTERS)

print("Running Hamiltonian sweeps...")
for s in range(N_SWEEPS):
    state = hamiltonian_sweep(state)
    print(f"  sweep {s+1}/{N_SWEEPS}")

# ============================================================
# Build cluster assignment matrix (JAX-friendly)
# ============================================================

S_wc = jax.nn.one_hot(state, N_CLUSTERS)        # (V, C)
counts = jnp.sum(S_wc, axis=0) + 1e-6

# Initial cluster latents
Z = (S_wc.T @ X) / counts[:, None]

# ============================================================
# Cluster-to-cluster causal weights
# ============================================================

J_cc = jnp.zeros((N_CLUSTERS, N_CLUSTERS))
for (i, j), w in cooc.items():
    J_cc = J_cc.at[state[i], state[j]].add(w)

# ============================================================
# Predictive coding free energy (VECTORISED)
# ============================================================

@jax.jit
def free_energy(Z):
    # prediction error
    X_hat = S_wc @ Z                      # (V, D)
    F_pc = jnp.sum((X - X_hat) ** 2)

    # causal Hamiltonian prior
    F_h = -jnp.sum(J_cc * (Z @ Z.T))

    return F_pc + LAMBDA_H * F_h

grad_F = jax.jit(jax.grad(free_energy))

# ============================================================
# Predictive coding relaxation
# ============================================================

print("Running predictive-coding relaxation...")
for step in range(PC_STEPS):
    Z = Z - ETA_Z * grad_F(Z)
    if step % 10 == 0:
        print(f"  PC step {step}/{PC_STEPS}")

Z_np = np.array(Z)
state_np = np.array(state)

# ============================================================
# Interpretability output
# ============================================================

print("\n" + "-" * 80)
print("RELAXED CLUSTER TEMPLATES")
print("-" * 80)

for c in range(N_CLUSTERS):
    idx = np.where(state_np == c)[0]
    if len(idx) == 0:
        continue
    scores = np.dot(X[idx], Z_np[c])
    top = idx[np.argsort(-scores)[:TOP_K]]
    print(f"\nCluster {c}:")
    for i in top:
        print(f"  {vocab[i]}")

print("\nDone.")
