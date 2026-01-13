# Substitution-Based Parsing: Experimental Branch

**Branch**: `experimental/substitution-parsing`
**Status**: Active Research/Experimental
**Contributors**: Rob Freeman, Claude Code (Anthropic)

---

## 🚀 Get Started in 5 Minutes

Want to see the parser in action? Follow these steps:

### Step 1: Download the Corpus

Download the Cornell Movie Dialog Corpus:

```bash
# Option A: Using curl (Mac)
curl -L -O http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip

# Option B: Using wget (Linux)
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip

# Option C: Manual download
# Visit: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
# Download and unzip to the current directory
```

**Expected result**: You should now have a `cornell movie-dialogs corpus/` directory.

### Step 2: Extract Dialog Lines

The Cornell corpus uses a special format. Extract the dialog text:

```bash
python3 extract_cornell_dialogs.py
```

**Expected output**:
```
Extracting dialogs from cornell movie-dialogs corpus/movie_lines.txt...
✓ Extracted 304,446 dialog lines to dialog_corpus.txt
✓ Output file size: 16.3 MB
```

**Expected result**: You should now have `dialog_corpus.txt` (~16 MB).

### Step 3: Build the Unit Catalog

This extracts n-grams and their contexts from the corpus (~2-3 minutes):

```bash
python3 prototype_topdown_units.py --build-catalog
```

**Expected output**:
```
Building unit catalog from cornell movie-dialogs corpus/...
Total tokens: 9,035,582
Processing 1-grams...
Processing 2-grams...
Processing 3-grams...
Processing 4-grams...
Extracted 32,294 units (min_freq=20)
  1-grams: 8,234
  2-grams: 15,892
  3-grams: 6,234
  4-grams: 1,934
Saved catalog to unit_catalog.pkl
```

**Expected result**: You should now have `unit_catalog.pkl` (~81MB).

### Step 3: Run the Parser!

```bash
python3 hamiltonian_parser.py
```

**Expected output**:
```
"i lost my money yesterday"
└─ [SPLIT]
   ├─ [UNIT] "i lost"
   └─ [SPLIT]
      ├─ [UNIT] "my money"      ← Successfully identified!
      └─ [UNIT] "yesterday"
```

**Success!** The parser correctly identifies "my money" as a unit (not "lost my").

---

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
- **`check_catalog.py`, `test_units.py`** - Testing utilities

### Documentation

- **`docs/parsing_approach_summary.pdf`** - Comprehensive 11-page summary
  - Evolution of thinking
  - Concrete examples with "go out" polysemy
  - Implementation architecture comparison
  - Current technical challenges
  - Connection to Goertzel's Hamiltonian framework

## More Examples

### Try Your Own Sentences

```bash
# Parse a specific sentence
python3 hamiltonian_parser.py --sentence "did you go out with sarah"

# With debug output
python3 hamiltonian_parser.py --debug --sentence "should we go out on friday"

# Using the simpler prototype
python3 prototype_topdown_units.py
```

### Sample Results

```
"did you go out with sarah"
└─ [SPLIT]
   ├─ [SPLIT]
   │  ├─ [UNIT] "did you"
   │  └─ [UNIT] "go out"        ← Polysemous phrase
   └─ [SPLIT]
      ├─ [UNIT] "with"
      └─ [UNIT] "sarah"

"my money is gone"
└─ [SPLIT]
   ├─ [UNIT] "my money"          ← Correct possessive NP
   └─ [SPLIT]
      ├─ [UNIT] "is"
      └─ [UNIT] "gone"
```

## Troubleshooting

### "FileNotFoundError: dialog_corpus.txt"

**Problem**: The corpus hasn't been downloaded.

**Solution**: Follow Step 1 above to download the Cornell Movie Dialog Corpus.

### "FileNotFoundError: unit_catalog.pkl"

**Problem**: The unit catalog hasn't been built yet.

**Solution**: Run `python3 preprocess_corpus.py` (Step 2 above).

### "ModuleNotFoundError: No module named 'numpy'"

**Problem**: Missing dependencies.

**Solution**: Install required packages:
```bash
pip install numpy
```

## Advanced Usage

### Build with Different Corpus

```bash
# Use your own corpus (one sentence per line)
python3 preprocess_corpus.py --corpus my_corpus.txt --output my_catalog.pkl

# Then use it
python3 hamiltonian_parser.py --catalog my_catalog.pkl
```

### Adjust Parameters

```bash
# Lower minimum frequency threshold (finds more rare units)
python3 preprocess_corpus.py --min-freq 5

# Enable experimental expansion-based energy
python3 hamiltonian_parser.py --use-expansion

# Run with iterative refinement
python3 hamiltonian_parser.py --max-iters 3
```

### Inspect the Catalog

```bash
# Check what units are in the catalog
python3 check_catalog.py

# Test specific units
python3 test_units.py
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
4. **Neural approach**: Replace explicit expansion with learned predictor?

## Architecture Options

See `docs/parsing_approach_summary.pdf` for detailed discussion.

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
# Core (required)
pip install numpy

# Optional (for future neural extensions)
# pip install torch transformers
```

## Data

Uses **Cornell Movie Dialog Corpus**:
- 220,579 conversational exchanges
- 304,713 utterances
- Good for short, substitutable phrases
- Download: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

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

**Documentation**: See `docs/parsing_approach_summary.pdf` for detailed exposition.

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
