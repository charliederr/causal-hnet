# Substitution-Based Parsing: Experimental Branch

**Branch**: `experimental/substitution-parsing`
**Status**: Active Research/Experimental
**Contributors**: Rob Freeman, Claude Code (Anthropic)

## Overview

This branch explores an alternative parsing approach based on **global substitutability patterns** rather than local context windows. The key insight is that true syntactic units should be identified by maximizing their **bidirectional expansion**: the set of units that can substitute in similar contexts, where contexts themselves are also expanded to handle novel sentences.

## Motivation: The "my money" Problem

Consider: *"I lost my money yesterday"*

**Question**: Should the parser group "lost my" or "my money"?

**Intuition**: "my money" should win because:
- "my money" substitutes with "your keys", "his wallet", "the car" across diverse contexts
- "lost my" requires the determiner to change with the verb ("lost your", "found my")
- Local n-gram frequency alone can't distinguish these

**This branch**: Implements corpus-wide substitution analysis to resolve these cases.

## Key Concepts

### 1. Bidirectional Expansion
```
Initial: unit = "go out", context = ("did you", "with Sarah")

Context Expansion:          Unit Expansion:
→ "did you ___ with Helen"  → go out, hang out
→ "would you ___ with me"   → meet up, get together
→ "we ___ together"         → chat, connect
→ "let's ___ on Friday"

Score: |units| × log(|contexts|) → larger = lower energy
```

### 2. Compositionality Bonus
Multi-word units get energy bonus proportional to:
```
actual_freq / (component_freq₁ × component_freq₂ / normalization)
```

Higher ratio = unit appears together more than expected from independence.

### 3. Top-Down Parsing
Test spans as units BEFORE attempting to split them (unlike bottom-up approaches).

## Implementation Files

### Core Implementations (in order of development)

1. **`prototype_topdown_units.py`** - Initial working prototype
   - Frequency-based energy with compositionality bonus
   - Successfully finds multi-word units
   - ~300 lines of simple, working code
   - Good starting point for understanding the approach

2. **`bidir_simple.py`** - Simplified bidirectional expansion
   - Uses aggregated context patterns (Counter objects)
   - Scalable but limited similarity metrics
   - Demonstrates bidirectional expansion concept

3. **`hamiltonian_parser.py`** - Full implementation
   - Complete Hamiltonian framework
   - Two modes: frequency-based (stable) and expansion-based (experimental)
   - Iterative energy minimization
   - Relational energy between spans
   - Production-ready architecture

### Supporting Files

- **`preprocess_corpus.py`** - Build unit catalog from text corpus
- **`learn_roles_simple.py`** - Learn role templates (directional clustering)
- **`bidirectional_expansion_parser.py`** - Earlier full implementation (memory issues)
- **`check_catalog.py`, `test_units.py`** - Testing utilities

### Documentation

- **`pdfs/parsing_approach_summary.pdf`** - Comprehensive 11-page summary
  - Evolution of thinking
  - Concrete examples with "go out" polysemy
  - Implementation architecture comparison
  - Current technical challenges
  - Connection to Goertzel's Hamiltonian framework

- **`pdfs/dynamic_roles_and_context_first_spans.pdf`** - Earlier exploration

## Quick Start

### Build the Catalog

```bash
# Download Cornell Movie Dialogs Corpus first
python3 preprocess_corpus.py

# This creates unit_catalog.pkl (~81MB)
# Contains 32k units with aggregated context patterns
```

### Run the Parser

```bash
# Basic parsing (recommended starting point)
python3 hamiltonian_parser.py --max-iters 0

# With debug output
python3 hamiltonian_parser.py --debug --max-iters 0

# Single sentence
python3 hamiltonian_parser.py --sentence "i lost my money yesterday"

# With expansion-based energy (experimental)
python3 hamiltonian_parser.py --use-expansion
```

### Run the Prototype (simpler)

```bash
python3 prototype_topdown_units.py
```

## Results

The parser successfully identifies meaningful multi-word units:

```
"i lost my money yesterday"
└─ [SPLIT]
   ├─ [UNIT] "i lost"
   └─ [SPLIT]
      ├─ [UNIT] "my money"      ← Correct!
      └─ [UNIT] "yesterday"

"did you go out with sarah"
└─ [SPLIT]
   ├─ [SPLIT]
   │  ├─ [UNIT] "did you"
   │  └─ [UNIT] "go out"        ← Polysemous phrase
   └─ [SPLIT]
      ├─ [UNIT] "with"
      └─ [UNIT] "sarah"
```

## Current Status

### ✅ Working

- Top-down parsing with unit-first testing
- Compositionality bonus for multi-word units
- Energy-based split decisions
- Finds units like "my money", "go out", "did you", "the door"
- Iterative refinement framework (infrastructure ready)

### 🚧 In Development

- **Scalability**: Bidirectional expansion O(n²) - needs indexing or embeddings
- **Context similarity**: Jaccard on aggregated contexts gives false positives
- **Relational energy**: Currently simple heuristic, needs learned patterns
- **Iterative proposals**: Random modifications, need gradient-based or smarter heuristics

### 🤔 Open Questions

1. **Context representation**: Exact tuples vs. aggregated vs. embeddings?
2. **Pre-clustering vs. dynamic**: Compute expansion at parse time or offline?
3. **Averaging problem**: How to cluster contexts by meaning without blending?
4. **Neural approach**: Replace explicit expansion with learned predictor? (Option B in PDF)

## Architecture Options (from PDF)

### Option A: Explicit Expansion (current)
```python
# Pre-build catalog of units and contexts
unit_contexts = build_catalog(corpus)

# Parse time: expand both unit and context
expansion = bidirectional_expansion(unit, current_context)
```
- **Pro**: Interpretable, theoretically grounded
- **Con**: Scalability, similarity metric

### Option B: Neural Predictor
```python
# Learn to predict relations
edge_predictor = train_model(corpus)

# Parse time: dynamic prediction
relations = edge_predictor(span_embeddings, context)
```
- **Pro**: Handles novel contexts, fast at inference
- **Con**: Less interpretable, requires training infrastructure

### Option C: Hybrid (future)
Pre-cluster common patterns, expand dynamically for rare cases.

## Comparison to Parent Project

This branch differs from the main causal-hnet approach in:

1. **Focus**: Global substitutability vs. local causal edges
2. **Method**: Bidirectional expansion vs. mean-field inference
3. **Compositionality**: Explicit bonus vs. emergent from edges
4. **Top-down**: Unit-first testing vs. bottom-up aggregation

Both share the Hamiltonian energy minimization framework.

## Dependencies

```bash
# Core
pip install numpy

# Optional (for future neural extensions)
# pip install torch transformers
```

## Testing

```bash
# Run all test sentences
python3 hamiltonian_parser.py

# Test specific functionality
python3 test_single_span.py
python3 check_catalog.py
```

## Data

Uses **Cornell Movie Dialog Corpus**:
- 220,579 conversational exchanges
- 304,713 utterances
- Good for short, substitutable phrases

Preprocessed to `unit_catalog.pkl` with n-grams (n=1..4) and aggregated contexts.

## Future Directions

1. **Better similarity metrics**:
   - Embed contexts with sentence encoders
   - Structural features (POS patterns)
   - Learned similarity functions

2. **Scalability**:
   - Approximate nearest neighbor search (FAISS)
   - Hierarchical clustering of contexts
   - Sparse representations

3. **Dynamic prediction**:
   - Neural edge predictor
   - Attention-based context matching
   - End-to-end differentiable parsing

4. **Evaluation**:
   - Compare to treebank parses
   - Human judgments on unit quality
   - Downstream task performance

## Contact & Collaboration

This is active research code. Feedback, suggestions, and collaboration welcome!

**Documentation**: See `pdfs/parsing_approach_summary.pdf` for detailed exposition.

## Acknowledgments

A further development of the dynamic substitution parsing idea from:

https://patents.google.com/patent/US7392174B2/en

And:

Parsing Using a Grammar of Word Association Vectors - 2014 preprint - https://arxiv.org/abs/1403.2152.

In harmony with positional categories in:

A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture
https://openreview.net/pdf?id=hP4dxXvvNc8

Interpreted and related to causal coding by Ben Goertzel

And initially implemented by Charlie Derr
[causal-hnet](https://github.com/charliederr/causal-hnet).
