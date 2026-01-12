#!/usr/bin/env python3
"""
Simple role/template learning using directional clustering (NumPy only).
Generates q0 role distributions for the Step 4B parser.

This is a simplified version of Step 4A that works without JAX.
"""

import sys
import numpy as np
from collections import Counter, defaultdict

# Configuration
MIN_FREQ = 10
MAX_VOCAB = 10000
COOC_WINDOW = 6
N_CLUSTERS = 16
N_SWEEPS = 30
GAMMA_SIZE = 0.8          # size penalty (prevents giant clusters)
UPDATE_PROB = 0.25        # inertial update fraction
SEED = 42

print("=" * 80)
print("LEARNING ROLE TEMPLATES (Simple NumPy version)")
print("=" * 80)

# Load text
input_file = sys.argv[1] if len(sys.argv) > 1 else "dialog_corpus.txt"
print(f"\nReading: {input_file}")
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read().lower()

tokens = text.split()
T = len(tokens)
print(f"Total tokens: {T:,}")

# Build vocabulary
print("Building vocabulary...")
freq = Counter(tokens)
vocab_list = [w for w, c in freq.most_common() if c >= MIN_FREQ]
vocab_list = vocab_list[:MAX_VOCAB]
word_to_id = {w: i for i, w in enumerate(vocab_list)}
V = len(vocab_list)
print(f"Vocabulary size: {V:,}")

# Convert to IDs
tok_ids = np.array([word_to_id.get(w, -1) for w in tokens], dtype=np.int32)
tok_ids = tok_ids[tok_ids >= 0]  # remove OOV
print(f"Valid token IDs: {len(tok_ids):,}")

# Build directional co-occurrence
print(f"\nBuilding directional co-occurrence (window={COOC_WINDOW})...")
cooc = defaultdict(float)
for i in range(len(tok_ids)):
    wi = tok_ids[i]
    end = min(i + COOC_WINDOW + 1, len(tok_ids))
    for j in range(i + 1, end):
        wj = tok_ids[j]
        cooc[(wi, wj)] += 1.0

# Normalize by geometric mean of node degrees
norm_out = np.zeros(V, dtype=np.float64)
norm_in = np.zeros(V, dtype=np.float64)
for (i, j), v in cooc.items():
    norm_out[i] += v
    norm_in[j] += v

edges = []
for (i, j), v in cooc.items():
    w = v / np.sqrt(norm_out[i] * norm_in[j] + 1e-8)
    edges.append((i, j, w))

print(f"Directional edges: {len(edges):,}")

# Initialize random cluster assignments
print(f"\nInitializing {N_CLUSTERS} clusters...")
rng = np.random.default_rng(SEED)
state = rng.integers(0, N_CLUSTERS, size=V, dtype=np.int32)

# Clustering with balanced sweeps (inertial updates)
print(f"Running {N_SWEEPS} clustering sweeps...")
for sweep in range(N_SWEEPS):
    # Compute field from neighbors
    field = np.zeros((V, N_CLUSTERS), dtype=np.float64)
    for (i, j, w) in edges:
        field[i, state[j]] += w

    # Size penalty
    counts = np.bincount(state, minlength=N_CLUSTERS).astype(np.float64) + 1.0
    penalty = GAMMA_SIZE * np.log(counts)

    # Score and best assignment
    scores = field - penalty[None, :]
    best = np.argmax(scores, axis=1)

    # Inertial update: only update a fraction
    mask = rng.random(V) < UPDATE_PROB
    state = np.where(mask, best, state)

    # Progress
    if (sweep + 1) % 5 == 0 or sweep == 0:
        counts = np.bincount(state, minlength=N_CLUSTERS)
        nonzero = int((counts > 0).sum())
        print(f"  Sweep {sweep+1}/{N_SWEEPS}: {nonzero} non-empty clusters, sizes={counts.tolist()}")

# Compute cluster-cluster couplings
print("\nComputing cluster→cluster couplings...")
J_cc = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.float64)
for (wi, wj, w) in edges:
    ci = state[wi]
    cj = state[wj]
    J_cc[ci, cj] += w

# Normalize
if J_cc.max() > 0:
    J_cc = J_cc / J_cc.max()

# Show top couplings
print("\nTop 10 cluster→cluster couplings:")
pairs = [(J_cc[a, b], a, b) for a in range(N_CLUSTERS) for b in range(N_CLUSTERS) if a != b]
pairs.sort(reverse=True)
for val, a, b in pairs[:10]:
    print(f"  c{a:02d} → c{b:02d}  strength={val:.3f}")

# Convert to soft role distributions with smoothing
print("\nConverting to soft role distributions...")
smoothing = 0.05
q0 = np.zeros((V, N_CLUSTERS), dtype=np.float32)
for i in range(V):
    c = state[i]
    q0[i, :] = smoothing / N_CLUSTERS
    q0[i, c] = 1.0 - smoothing + (smoothing / N_CLUSTERS)

q0 = q0 / q0.sum(axis=-1, keepdims=True)

# Save outputs
print("\nSaving outputs...")
np.savez_compressed(
    "hnet_q0_learned.npz",
    q0=q0
)
print("  ✓ hnet_q0_learned.npz")

np.savez_compressed(
    "hnet_clusters.npz",
    state=state,
    J_cc=J_cc.astype(np.float32),
    vocab=np.array(vocab_list, dtype=object)
)
print("  ✓ hnet_clusters.npz")

print("\n" + "=" * 80)
print("DONE! Learned role templates saved.")
print("=" * 80)
print(f"\nRun the parser with:")
print(f"  python3 hdp1.py --K {N_CLUSTERS} --q0_path hnet_q0_learned.npz --num_segments 3 --segment_len 24")
print()
