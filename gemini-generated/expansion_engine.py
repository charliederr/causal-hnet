import jax
import jax.numpy as jnp
import numpy as np

# --- 1. CONFIGURATION & MOCK DATA ---
# In a real scenario, DIM would be 768 (BERT) or similar.
EMBEDDING_DIM = 64 
NUM_CONTEXT_CLUSTERS = 10  # Compressed from millions of raw contexts
NUM_UNITS = 6              # e.g., "go out", "my money", "lost my"

# Labels for readability in this demo
UNIT_LABELS = [
    "go out",       # 0: Polysemous (social + motion)
    "hang out",     # 1: Social only
    "meet up",      # 2: Social only
    "exit",         # 3: Motion only
    "leave",        # 4: Motion only
    "my money"      # 5: Unrelated
]

# We simulate "Context Centroids" (Matrix C). 
# Rows = Cluster Centroids, Cols = Embedding Dimensions
# We create random vectors to represent different "meaning spaces"
key = jax.random.PRNGKey(0)
context_centroids = jax.random.normal(key, (NUM_CONTEXT_CLUSTERS, EMBEDDING_DIM))

# We simulate the Adjacency Matrix (Matrix A).
# Rows = Units, Cols = Context Clusters.
# A[i, j] = 1 if Unit i appears in Context Cluster j.
# Let's manually define relations to test the "social vs motion" logic[cite: 114, 126].
adj_matrix = np.zeros((NUM_UNITS, NUM_CONTEXT_CLUSTERS))

# Assume Clusters 0,1,2 are "Social" contexts ("did you ___ with Sarah")
adj_matrix[0, 0:3] = 1.0  # "go out" fits social
adj_matrix[1, 0:3] = 1.0  # "hang out" fits social
adj_matrix[2, 0:3] = 1.0  # "meet up" fits social

# Assume Clusters 3,4,5 are "Motion" contexts ("I ___ the door")
adj_matrix[0, 3:6] = 1.0  # "go out" fits motion (POLYSEMY!)
adj_matrix[3, 3:6] = 1.0  # "exit" fits motion
adj_matrix[4, 3:6] = 1.0  # "leave" fits motion

# Assume Clusters 6,7 are "Possessive" contexts ("I lost ___")
adj_matrix[5, 6:8] = 1.0  # "my money" fits

# Convert to JAX array
adj_matrix = jnp.array(adj_matrix)

# --- 2. JAX ENGINE IMPLEMENTATION ---

@jax.jit
def get_similarity_mask(input_vec, centroids, temperature=1.0):
    """
    Step 2.1: Context Expansion [cite: 42]
    Computes cosine similarity between input vector and all known context centroids.
    Returns a 'soft mask' (probability distribution) over context clusters.
    """
    # Normalize input and centroids for cosine similarity
    input_norm = input_vec / jnp.linalg.norm(input_vec)
    centroids_norm = centroids / jnp.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Dot product: (1, D) @ (D, K) -> (1, K)
    # This measures how close the current context is to every known cluster.
    similarities = jnp.dot(centroids_norm, input_norm)
    
    # Softmax to create a probability mask (expands to similar contexts)
    mask = jax.nn.softmax(similarities / temperature)
    return mask

@jax.jit
def propagate_to_units(context_mask, adjacency_matrix):
    """
    Step 2.2: Unit Expansion [cite: 56]
    Finds units that fit into the expanded contexts.
    """
    # Matrix vector multiplication: (Units, Clusters) @ (Clusters, 1) -> (Units, 1)
    # Result: How strongly each unit is activated by the context expansion.
    unit_activations = jnp.dot(adjacency_matrix, context_mask)
    return unit_activations

@jax.jit
def compute_energy(unit_activations, context_mask):
    """
    Step 2.3: Energy Score 
    E = -log(|units| * log(|contexts| + 1))
    
    We use 'soft counts' (sum of probabilities) instead of hard set sizes.
    """
    # Soft count of expanded units (sum of activations)
    unit_volume = jnp.sum(unit_activations)
    
    # Soft count of expanded contexts (entropy-like or simple sum of significant weights)
    # For now, we use the effective number of clusters involved (inverse participation ratio or similar)
    # But strictly following the paper's formula with soft sums:
    context_volume = jnp.sum(context_mask) # This will be 1.0 due to softmax, need unnormalized or thresholded?
    
    # To capture "Breadth", we might want the entropy of the mask, 
    # but let's stick to the paper's intuition: Volume of the Expansion.
    # If we use Softmax, sum is 1. Let's use the raw similarities > threshold logic implicitly
    # by using the magnitude of activations.
    
    # Adjusted Logic: A "strong" unit has high activation sum.
    # E = -log(unit_volume) (Simplified Hamiltonian)
    energy = -jnp.log(unit_volume + 1e-9) 
    return energy

@jax.jit
def run_expansion_pass(context_vec, centroids, adj_mat):
    """
    Runs the full bidirectional expansion for a single query.
    """
    # 1. Expand Contexts (Where else could this occur?)
    ctx_mask = get_similarity_mask(context_vec, centroids, temperature=0.1)
    
    # 2. Expand Units (Who else fits there?)
    unit_acts = propagate_to_units(ctx_mask, adj_mat)
    
    # 3. Score
    score = compute_energy(unit_acts, ctx_mask)
    
    return unit_acts, ctx_mask, score

# --- 3. DEMONSTRATION ---

def demo():
    print("--- JAX Bidirectional Expansion Demo ---")
    
    # Case A: Simulate a "Social" context input
    # (We cheat and pick a vector close to Cluster 0)
    social_input = context_centroids[0] + jax.random.normal(key, (EMBEDDING_DIM,)) * 0.1
    
    print("\n[Input]: 'Did you ___ with Sarah?' (Social Context)")
    units, contexts, energy = run_expansion_pass(social_input, context_centroids, adj_matrix)
    
    print(f"Energy Score: {energy:.4f} (Lower is better)")
    print("Top Activated Units:")
    top_k_indices = jnp.argsort(units)[::-1][:3]
    for idx in top_k_indices:
        print(f"  - {UNIT_LABELS[idx]}: {units[idx]:.4f}")

    # Case B: Simulate a "Motion" context input
    # (Pick a vector close to Cluster 3)
    motion_input = context_centroids[3] + jax.random.normal(key, (EMBEDDING_DIM,)) * 0.1
    
    print("\n[Input]: 'I ___ the door' (Motion Context)")
    units, contexts, energy = run_expansion_pass(motion_input, context_centroids, adj_matrix)
    
    print(f"Energy Score: {energy:.4f}")
    print("Top Activated Units:")
    top_k_indices = jnp.argsort(units)[::-1][:3]
    for idx in top_k_indices:
        print(f"  - {UNIT_LABELS[idx]}: {units[idx]:.4f}")

    # Case C: Polysemy Check [cite: 138]
    # Note how 'go out' (Index 0) should appear in BOTH lists, 
    # but its 'peers' (hang out vs exit) change based on the input vector.

if __name__ == "__main__":
    demo()
