# Substitution-Based Sequence Prediction

## The Transformer Analogy

This branch explores using **substitution-based parsing** as a foundation for sequence prediction, analogous to how transformers work but with key differences.

### What Transformers Do

1. **Static embeddings**: Each token has a learned vector
2. **Attention**: Similar embeddings attend to each other
3. **Prediction**: Final hidden state predicts next token

The key insight: **tokens with similar embeddings predict similar continuations**.

### What Substitution-Based Prediction Does

1. **Dynamic context signatures**: Each position gets a signature computed on-the-fly from its surrounding context
2. **Context overlap**: Positions with similar context signatures are substitutable
3. **Prediction**: Find units with overlapping contexts, aggregate their typical continuations

The key insight is the same: **tokens with similar contexts predict similar continuations** - but discovered dynamically rather than learned.

## The Connection to CEBS

This work connects to Ben Goertzel's CEBS (Contextual Energy-Based Slotting) paper:

| CEBS (Visual) | This Work (Linguistic) | Transformers |
|---------------|------------------------|--------------|
| Entity = context overlap class | Unit = context overlap class | Token = embedding vector |
| Context signatures via message passing | Context signatures via aggregation | Embeddings via lookup + attention |
| Binding = entity-slot assignment | Parsing = unit-span assignment | Attention weights |
| Dynamic, computed per-input | Dynamic, computed per-input | Static, learned once |

## How Prediction Works

```
Given prompt P = [t_1, ..., t_n], predict t_{n+1}:

1. Compute context signature at position n:
   C_n = {left context words with counts}

2. Find units with overlapping left-contexts:
   matching = {u : overlap(C_n, left_context(u)) > θ}

3. Aggregate right-context predictions:
   P(t_{n+1}) ∝ Σ_u weight(u) × right_context(u)[t_{n+1}]

4. Weight by:
   - Context overlap strength
   - Unit frequency (reliability)
   - Hierarchy level (if using parse structure)
```

## Hierarchical Prediction

The prompt can be parsed into a hierarchy of units:

```
"i want to go"
└─ [SPLIT] "i want to go"
   ├─ [UNIT] "i"
   └─ [SPLIT] "want to go"
      ├─ [UNIT] "want"
      └─ [SPLIT] "to go"
         ├─ [UNIT] "to"
         └─ [UNIT] "go"
```

Each level contributes predictions:
- **Higher levels** = more abstract patterns (sentence-level)
- **Lower levels** = more specific patterns (word-level)

The "central/strongest prediction" comes from combining these levels, weighted by hierarchy.

## Current Status

The initial implementation demonstrates the mechanism:
- ✅ Context signatures computed dynamically
- ✅ Context overlap finds matching units
- ✅ Right-context aggregation produces predictions
- ✅ Hierarchical prediction works

But predictions need improvement:
- ❌ Context matching too loose (finds too many false positives)
- ❌ Weighting scheme too simple
- ❌ Generation lacks coherence

## Next Steps

1. **Tighten context matching**
   - Use weighted Jaccard (by frequency)
   - Require minimum overlap threshold
   - Consider position in context (immediate vs distant)

2. **Better weighting**
   - Weight by specificity (rare contexts more informative)
   - Weight by recency (recent context more relevant)
   - Learn weights from prediction accuracy

3. **Coherent generation**
   - Beam search with coherence scoring
   - Use parse structure to constrain generation
   - Enforce grammatical patterns from units

4. **Comparison with transformers**
   - Evaluate on perplexity
   - Compare interpretability
   - Analyze what each method captures

## Running the Code

```bash
# Build unit catalog (if not already done)
python3 bidir_simple.py

# Run predictor
python3 substitution_predictor.py
```

## References

- CEBS paper: Goertzel, "Contextual Energy-Based Slotting for Visual Tokenization" (v5)
- Related: The parsing work explores similar ideas for linguistic structure
