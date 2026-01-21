#!/usr/bin/env python3
"""
Substitution-Based Sequence Predictor

Uses dynamic context signatures (from substitution parsing) as an alternative
to static transformer embeddings. Predicts next tokens based on context overlap
with units in the catalog.

Key insight: Units with similar left-contexts predict similar right-contexts.
This is the same principle that transformers learn, but computed dynamically
from substitution patterns rather than learned weights.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import math
import pickle
import heapq

# Import from existing parser
from bidir_simple import (
    UnitCatalog, ContextPattern, Span, ParseNode,
    context_similarity_aggregated, SimpleBidirParser, print_tree
)


@dataclass
class Prediction:
    """A predicted token with score and explanation."""
    token: str
    score: float
    contributing_units: List[Tuple[str, float]]  # (unit_text, contribution)


@dataclass
class ContextSignature:
    """
    Dynamic 'embedding' computed from context.
    Unlike static embeddings, this is computed on-the-fly from the input.
    """
    left_words: Counter
    right_words: Counter  # What we're trying to predict
    hierarchy_level: int
    source_units: List[str]  # Units that contributed to this signature

    def overlap_with(self, other: 'ContextSignature') -> float:
        """Compute context overlap (analogous to embedding similarity)."""
        return context_similarity_aggregated(self.left_words, other.left_words)


class SubstitutionPredictor:
    """
    Predicts next tokens using substitution-based context signatures.

    The key analogy to transformers:
    - Transformer: learned embeddings relate tokens by similarity
    - This: dynamic context signatures relate tokens by substitutability

    Both discover that elements with similar contexts predict similar continuations.
    """

    def __init__(self, catalog: UnitCatalog,
                 context_window: int = 5,
                 top_k_context: int = 20,
                 min_unit_freq: int = 5):
        self.catalog = catalog
        self.context_window = context_window
        self.top_k_context = top_k_context
        self.min_unit_freq = min_unit_freq

        # Build reverse index: left_word -> units that have this left context
        self._build_indices()

    def _build_indices(self):
        """Build indices for fast lookup."""
        print("Building prediction indices...")

        # Index: context word -> units with that context
        self.left_context_index = defaultdict(set)
        self.right_context_index = defaultdict(set)

        # Store right-context distributions for prediction
        self.unit_right_contexts = {}

        # Filter to frequent units only for speed
        freq_units = {
            text: pattern for text, pattern in self.catalog.units.items()
            if pattern.count >= self.min_unit_freq
        }
        print(f"  Filtering to {len(freq_units):,} units with freq >= {self.min_unit_freq}")

        for i, (unit_text, pattern) in enumerate(freq_units.items()):
            if i % 5000 == 0:
                print(f"  Processing unit {i:,}...")

            # Index by top left context words only
            for word, count in pattern.left_words.most_common(self.top_k_context):
                self.left_context_index[word].add(unit_text)

            # Store normalized right-context distribution
            total = sum(pattern.right_words.values())
            if total > 0:
                self.unit_right_contexts[unit_text] = {
                    word: count / total
                    for word, count in pattern.right_words.most_common(30)
                }

        print(f"  Indexed {len(self.unit_right_contexts):,} units")
        print(f"  Left context vocabulary: {len(self.left_context_index):,} words")

    def compute_context_signature(self, tokens: List[str],
                                   position: int) -> ContextSignature:
        """
        Compute the dynamic 'embedding' at a position.
        This aggregates context from the tokens preceding this position.
        """
        # Get left context
        start = max(0, position - self.context_window)
        left_tokens = tokens[start:position]
        left_words = Counter(left_tokens)

        # Get right context (what we have so far, for partial matching)
        end = min(len(tokens), position + self.context_window)
        right_tokens = tokens[position:end]
        right_words = Counter(right_tokens)

        return ContextSignature(
            left_words=left_words,
            right_words=right_words,
            hierarchy_level=0,
            source_units=[]
        )

    def find_matching_units(self, context_sig: ContextSignature,
                           threshold: float = 0.1,
                           max_units: int = 100) -> List[Tuple[str, float]]:
        """
        Find units whose left-context overlaps with current context.
        Returns: [(unit_text, overlap_score), ...]
        """
        # Find candidate units via index
        candidates = set()
        for word in context_sig.left_words:
            candidates.update(self.left_context_index.get(word, set()))

        # Score candidates by context overlap
        scored = []
        for unit_text in candidates:
            pattern = self.catalog.get_unit(unit_text)
            if not pattern:
                continue

            overlap = context_similarity_aggregated(
                context_sig.left_words,
                pattern.left_words
            )

            if overlap >= threshold:
                scored.append((unit_text, overlap))

        # Sort by overlap score
        scored.sort(key=lambda x: -x[1])
        return scored[:max_units]

    def predict_next(self, tokens: List[str],
                     top_k: int = 10,
                     temperature: float = 1.0) -> List[Prediction]:
        """
        Predict next token(s) given a sequence.

        This is the core prediction method:
        1. Compute context signature at end of sequence
        2. Find units with overlapping left-contexts
        3. Aggregate their right-contexts as predictions
        """
        if not tokens:
            return []

        # 1. Compute context signature at prediction position
        context_sig = self.compute_context_signature(tokens, len(tokens))

        # 2. Find matching units
        matching_units = self.find_matching_units(context_sig, threshold=0.05)

        if not matching_units:
            return [Prediction("<unk>", 1.0, [])]

        # 3. Aggregate predictions from matching units
        prediction_scores = defaultdict(float)
        prediction_sources = defaultdict(list)

        for unit_text, overlap_score in matching_units:
            right_dist = self.unit_right_contexts.get(unit_text, {})

            # Weight by overlap and unit frequency
            pattern = self.catalog.get_unit(unit_text)
            freq_weight = math.log(pattern.count + 1) if pattern else 1.0
            weight = overlap_score * freq_weight

            for word, prob in right_dist.items():
                contribution = weight * prob
                prediction_scores[word] += contribution
                prediction_sources[word].append((unit_text, contribution))

        # 4. Normalize and return top-k
        total = sum(prediction_scores.values())
        if total == 0:
            return [Prediction("<unk>", 1.0, [])]

        predictions = []
        for word, score in sorted(prediction_scores.items(),
                                   key=lambda x: -x[1])[:top_k]:
            normalized_score = score / total
            # Apply temperature
            if temperature != 1.0:
                normalized_score = normalized_score ** (1.0 / temperature)

            predictions.append(Prediction(
                token=word,
                score=normalized_score,
                contributing_units=sorted(prediction_sources[word],
                                         key=lambda x: -x[1])[:5]
            ))

        return predictions

    def predict_with_hierarchy(self, tokens: List[str],
                               parser: SimpleBidirParser,
                               top_k: int = 10) -> List[Prediction]:
        """
        Predict using hierarchical structure.

        The prompt is parsed into a hierarchy of units, and predictions
        come from multiple levels - abstract (high-level units) and
        specific (low-level/local patterns).
        """
        # Parse the prompt
        if len(tokens) < 2:
            return self.predict_next(tokens, top_k)

        tree = parser.parse(tokens)

        # Collect units at different hierarchy levels
        hierarchy_units = self._collect_hierarchy_units(tree, tokens)

        # Aggregate predictions from all levels
        prediction_scores = defaultdict(float)
        prediction_sources = defaultdict(list)

        for level, units_at_level in hierarchy_units.items():
            # Higher levels get more weight (more abstract patterns)
            level_weight = 1.0 + 0.5 * level

            for unit_text, span_info in units_at_level:
                pattern = self.catalog.get_unit(unit_text)
                if not pattern:
                    continue

                right_dist = self.unit_right_contexts.get(unit_text, {})
                freq_weight = math.log(pattern.count + 1)

                for word, prob in right_dist.items():
                    contribution = level_weight * freq_weight * prob
                    prediction_scores[word] += contribution
                    prediction_sources[word].append(
                        (f"L{level}:{unit_text}", contribution)
                    )

        # Normalize
        total = sum(prediction_scores.values())
        if total == 0:
            return self.predict_next(tokens, top_k)  # Fallback

        predictions = []
        for word, score in sorted(prediction_scores.items(),
                                   key=lambda x: -x[1])[:top_k]:
            predictions.append(Prediction(
                token=word,
                score=score / total,
                contributing_units=sorted(prediction_sources[word],
                                         key=lambda x: -x[1])[:5]
            ))

        return predictions

    def _collect_hierarchy_units(self, node: ParseNode,
                                  tokens: List[str],
                                  level: int = 0) -> Dict[int, List[Tuple[str, dict]]]:
        """Collect units from parse tree at each hierarchy level."""
        result = defaultdict(list)

        if node.is_leaf():
            # This is a unit
            result[level].append((node.span.text, {
                'start': node.span.start,
                'end': node.span.end,
                'energy': node.energy
            }))
        else:
            # Recurse into children
            if node.left:
                for lvl, units in self._collect_hierarchy_units(
                    node.left, tokens, level + 1
                ).items():
                    result[lvl].extend(units)
            if node.right:
                for lvl, units in self._collect_hierarchy_units(
                    node.right, tokens, level + 1
                ).items():
                    result[lvl].extend(units)

        return result

    def generate(self, prompt: List[str],
                 max_tokens: int = 10,
                 temperature: float = 1.0,
                 use_hierarchy: bool = False,
                 parser: Optional[SimpleBidirParser] = None) -> List[str]:
        """
        Generate a continuation of the prompt.
        """
        tokens = list(prompt)

        for _ in range(max_tokens):
            if use_hierarchy and parser:
                predictions = self.predict_with_hierarchy(tokens, parser)
            else:
                predictions = self.predict_next(tokens, top_k=10,
                                               temperature=temperature)

            if not predictions or predictions[0].token == "<unk>":
                break

            # Sample from predictions (or take top-1 for greedy)
            if temperature == 0:
                next_token = predictions[0].token
            else:
                # Simple weighted sampling
                import random
                weights = [p.score for p in predictions]
                total = sum(weights)
                r = random.random() * total
                cumsum = 0
                next_token = predictions[0].token
                for p in predictions:
                    cumsum += p.score
                    if r <= cumsum:
                        next_token = p.token
                        break

            tokens.append(next_token)

            # Stop on end-of-sentence markers
            if next_token in ['.', '?', '!', '<eos>']:
                break

        return tokens[len(prompt):]


def main():
    print("=" * 80)
    print("SUBSTITUTION-BASED SEQUENCE PREDICTOR")
    print("=" * 80)
    print()
    print("Key insight: Context overlap determines substitutability,")
    print("and substitutable units predict similar continuations.")
    print("This is what transformers learn - but computed dynamically.")
    print("=" * 80)

    # Load catalog
    catalog = UnitCatalog()
    try:
        catalog.load('unit_catalog.pkl')
    except FileNotFoundError:
        print("Error: unit_catalog.pkl not found. Run bidir_simple.py first.")
        return

    # Create predictor
    predictor = SubstitutionPredictor(catalog)

    # Create parser for hierarchical prediction
    parser = SimpleBidirParser(catalog, debug=False)

    print("\n" + "=" * 80)
    print("PREDICTION EXAMPLES")
    print("=" * 80)

    test_prompts = [
        ["i", "want", "to"],
        ["do", "you", "want", "to"],
        ["she", "said", "that"],
        ["we", "should", "go"],
        ["the", "cat", "sat", "on", "the"],
        ["i", "don't", "know", "what"],
    ]

    for prompt in test_prompts:
        print(f"\n{'-' * 60}")
        print(f"PROMPT: \"{' '.join(prompt)}\"")
        print('-' * 60)

        # Basic prediction
        predictions = predictor.predict_next(prompt, top_k=5)
        print("\nTop predictions (context-overlap method):")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. '{pred.token}' (score: {pred.score:.3f})")
            if pred.contributing_units:
                top_unit = pred.contributing_units[0]
                print(f"      ← from unit '{top_unit[0]}'")

        # Generate continuation
        print(f"\nGenerated continuation:")
        continuation = predictor.generate(prompt, max_tokens=5, temperature=0.8)
        full_text = ' '.join(prompt + continuation)
        print(f"  \"{full_text}\"")

    print("\n" + "=" * 80)
    print("HIERARCHICAL PREDICTION EXAMPLE")
    print("=" * 80)

    prompt = ["i", "want", "to", "go"]
    print(f"\nPROMPT: \"{' '.join(prompt)}\"")

    print("\nParse tree:")
    tree = parser.parse(prompt)
    print_tree(tree)

    print("\nHierarchical predictions:")
    predictions = predictor.predict_with_hierarchy(prompt, parser, top_k=5)
    for i, pred in enumerate(predictions, 1):
        print(f"  {i}. '{pred.token}' (score: {pred.score:.3f})")
        for source in pred.contributing_units[:2]:
            print(f"      ← {source[0]}")


if __name__ == "__main__":
    main()
