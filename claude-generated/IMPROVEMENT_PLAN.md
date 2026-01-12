# Improvement Plan: causal_attention_multihead.py

Based on analysis of `docs/Causal-HNet_v1.pdf` (Goertzel, Dec 2025)

## Current State

The existing implementation provides:
- Multi-head attention with template biases
- Learnable template prototypes (Zc) via gradient descent
- Soft template assignments via softmax
- Template compatibility matrix W_psi (cosine similarity)
- Directional coupling matrix J_pmi (from co-occurrence)
- Entropy regularization to prevent template collapse

## Gap Analysis

| Paper Section | Paper Concept | Current Code | Gap |
|---------------|---------------|--------------|-----|
| 2.1 | Predictive-coding free energy F_PC | None | Full implementation needed |
| 2.2 | Context-gated modular updates | None | Full implementation needed |
| 3.1 | Hamiltonian energies E_r(x,y)=0 when satisfied | Uses similarity scores | Need true "zero-when-satisfied" energies |
| 4.1 | Soft node states p = σ(u/T) | softmax over templates | OK but could add temperature annealing |
| 4.2 | Binary-consistent pseudo-Boolean extension | None | Need Eq. 9 formulation |
| 4.3 | Binarization regularizer F_bin | None | Need β·Σp_i(1-p_i) |
| 5.1 | Soft edge masks m_ij(u) | None | Need learned edge gating |
| 5.1 | Relation-type mixtures w_ij,r(u) | None | Need mixture predictor |
| 5.2 | Relational energy F_rel | None | Need Eq. 11 |
| 6 | Combined F_total = F_PC + λF_rel + F_bin | None | Need unified objective |
| 7 | Sparse candidate pairs for attention | local_window only | Need top-k sparse selection |

## Implementation Plan

### Phase 1: Hamiltonian Energy Foundation (Priority: HIGH)

**1.1 Define Relation Types and H_r Matrices**
- Create learnable 2×2 Hamiltonian matrices H_r for each relation type r ∈ R
- Include offset k_r to ensure E_r=0 when constraint satisfied
- Start with |R|=4 relation types (can expand later)

**1.2 Implement Binary-Consistent Energy Extension**
```python
def relaxed_pair_energy(p, q, H_r, k_r):
    """Eq. 9: E_r(p,q) = h11*p + (h12+h21)*p*q + h22*q + k_r"""
    h11, h12, h21, h22 = H_r[0,0], H_r[0,1], H_r[1,0], H_r[1,1]
    return h11 * p + (h12 + h21) * p * q + h22 * q + k_r
```

**1.3 Add Binarization Regularizer**
```python
def binarization_loss(p, beta=0.1):
    """Eq. 10: Encourages p toward {0,1}"""
    return beta * jnp.sum(p * (1 - p))
```

### Phase 2: Relational Energy Module (Priority: HIGH)

**2.1 Soft Edge Mask Predictor**
- Neural head that predicts m_ij(u) ∈ [0,1] for candidate edges
- Input: concatenated/differenced latent representations

**2.2 Relation-Type Mixture Predictor**
- Neural head that predicts w_ij,r(u) with softmax over R
- Allows edges to express mixture of relation types

**2.3 Full Relational Energy F_rel**
```python
def relational_energy(p, m, w, H, k, edges):
    """Eq. 11: F_rel = Σ_{(i,j)∈E0} m_ij Σ_r w_ij,r E_r(p_i, p_j)"""
    total = 0.0
    for (i, j) in edges:
        edge_energy = 0.0
        for r in range(num_relations):
            edge_energy += w[i,j,r] * relaxed_pair_energy(p[i], p[j], H[r], k[r])
        total += m[i,j] * edge_energy
    return total
```

### Phase 3: Predictive-Coding Framework (Priority: MEDIUM)

**3.1 Multi-Layer Latent Representation**
- Add latent states z^(ℓ) for L layers
- Define layer-wise prediction functions f_ℓ

**3.2 Predictive-Coding Energy F_PC**
```python
def predictive_coding_energy(x, z, f, sigma):
    """Eq. 1: F_PC = Σ (1/2σ²)||z^(ℓ-1) - f_ℓ(z^(ℓ))||²"""
    total = 0.0
    z_prev = x  # z^(0) = x
    for ell in range(L):
        pred = f[ell](z[ell])
        total += 0.5 / (sigma[ell]**2) * jnp.sum((z_prev - pred)**2)
        z_prev = z[ell]
    return total
```

**3.3 Latent Inference Loop**
- Iterative gradient descent on z to minimize F_total
- Configurable number of inference steps

### Phase 4: Context-Gated Modules (Priority: MEDIUM)

**4.1 Module Partitioning**
- Partition θ into K modules (e.g., by attention head, by layer)

**4.2 Context Gates**
- Binary gates g_k,c ∈ {0,1} per module per context
- Could use learned gating or rule-based

**4.3 Gated Parameter Updates**
```python
def gated_update(params, grads, gates, lr):
    """Eq. 4: θ_k ← θ_k - η·g_k,c·∇θ_k F"""
    return [p - lr * g * grad for p, g, grad in zip(params, gates, grads)]
```

### Phase 5: Sparse Attention Integration (Priority: LOW)

**5.1 Sparse Candidate Selection**
- For each token i, select C(i) = top-k candidates by base attention logits
- Apply Hamiltonian bias only to (i, j) where j ∈ C(i)

**5.2 Hamiltonian Attention Bias**
```python
def hamiltonian_attention_bias(logits, p, e, gamma, C):
    """Eq. 13: ℓ_ij ← ℓ_ij - γ·E_pair(p_i, p_j, e_ij) for j ∈ C(i)"""
    # Only compute for sparse candidate pairs
    for i in range(S):
        for j in C[i]:
            logits[i,j] -= gamma * pair_energy(p[i], p[j], e[i,j])
    return logits
```

---

## Recommended Next Step

**Implement Phase 1.1-1.3: Hamiltonian Energy Foundation**

Create a new file `causal_attention_hamiltonian.py` that extends the current implementation with:

1. `HamiltonianRelation` class with learnable H_r matrices
2. `relaxed_pair_energy()` function (Eq. 9)
3. `binarization_loss()` function (Eq. 10)
4. Modified `ModelParams` to include H_r matrices
5. Integration into the attention bias computation

This foundational change enables all subsequent phases and directly addresses the paper's core contribution: treating relational constraints as differentiable energy terms.

### Concrete Changes to Make:

```python
# Add to ModelParams
class ModelParams(NamedTuple):
    Zc: jnp.ndarray      # Template prototypes: (n_clusters, emb_dim)
    Wq: jnp.ndarray      # Query projections
    Wk: jnp.ndarray      # Key projections
    Wv: jnp.ndarray      # Value projections
    Wo: jnp.ndarray      # Output projection
    # NEW: Hamiltonian relation parameters
    H_r: jnp.ndarray     # Relation matrices: (n_relations, 2, 2)
    k_r: jnp.ndarray     # Relation offsets: (n_relations,)

# Add new functions
def init_hamiltonian_relations(key, n_relations):
    """Initialize H_r matrices to encode useful default relations."""
    # Examples: IMPLIES (1,0)->0, XOR, AND, etc.
    ...

def relaxed_pair_energy(p_i, p_j, H_r, k_r):
    """Binary-consistent energy extension (Eq. 9)"""
    ...

def compute_hamiltonian_attention_bias(p, H_r, k_r, w_r):
    """Replace tpl_bias with true Hamiltonian energies"""
    ...
```

---

## Success Criteria

Phase 1 complete when:
- [ ] H_r matrices are learnable parameters
- [ ] relaxed_pair_energy matches Eq. 9 exactly
- [ ] Binarization regularizer is added to training loss
- [ ] Attention bias uses Hamiltonian energies instead of simple similarity
- [ ] Model still trains and produces interpretable template flows
