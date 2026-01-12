#!/usr/bin/env python3
"""
Hamiltonian Dynamic Parser

Implements the full Hamiltonian framework from Goertzel's paper:
- Span embeddings via expansion sets (not averaging)
- Relational energy terms between spans
- Iterative energy minimization
- Dynamic split prediction

Key differences from bidirectional prototypes:
1. Expansion sets ARE the embeddings (no separate representation)
2. Relational energy between span pairs (not just individual spans)
3. Iterative refinement loop
4. Joint optimization of all spans simultaneously
"""

import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict
import math
import pickle

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContextPattern:
    """Aggregated context distribution for a unit."""
    left_words: Counter
    right_words: Counter
    count: int

@dataclass
class Span:
    """A substring being parsed."""
    start: int
    end: int
    tokens: List[str]

    @property
    def text(self):
        return " ".join(self.tokens)

    @property
    def length(self):
        return self.end - self.start

    def __hash__(self):
        return hash((self.start, self.end, tuple(self.tokens)))

    def __eq__(self, other):
        return (self.start == other.start and
                self.end == other.end and
                self.tokens == other.tokens)

@dataclass
class SpanState:
    """Dynamic state for a span during parsing."""
    span: Span

    # Expansion sets (these ARE the embedding)
    unit_expansion: Set[str] = field(default_factory=set)
    context_expansion: Set[frozenset] = field(default_factory=set)

    # Energy components
    unit_energy: float = float('inf')
    split_energy: float = float('inf')
    relational_energy: float = 0.0

    # Parsing decisions
    is_unit: bool = False  # True if treated as unit, False if split
    split_point: Optional[int] = None
    left_child: Optional['SpanState'] = None
    right_child: Optional['SpanState'] = None

    @property
    def energy(self) -> float:
        """Return the actual energy of this span (unit or split)."""
        return self.unit_energy if self.is_unit else self.split_energy

# =============================================================================
# UNIT CATALOG (reuse from bidir_simple)
# =============================================================================

class UnitCatalog:
    """Catalog of units with aggregated context patterns."""

    def __init__(self):
        self.units = {}  # text -> ContextPattern
        self.expansion_cache = {}  # (text, context_hash) -> (unit_exp, ctx_exp)

    def load(self, path: str):
        print(f"Loading catalog from {path}...")
        with open(path, 'rb') as f:
            self.units = pickle.load(f)
        print(f"Loaded {len(self.units):,} units")

    def get_unit(self, text: str) -> Optional[ContextPattern]:
        return self.units.get(text)

    def has_unit(self, text: str) -> bool:
        return text in self.units

# =============================================================================
# CONTEXT SIMILARITY
# =============================================================================

def context_similarity_aggregated(ctx1_counter: Counter,
                                 ctx2_counter: Counter) -> float:
    """
    Similarity between aggregated context distributions.
    Uses Jaccard on top-k words.
    """
    if not ctx1_counter or not ctx2_counter:
        return 0.0

    top1 = set(w for w, c in ctx1_counter.most_common(10))
    top2 = set(w for w, c in ctx2_counter.most_common(10))

    if not top1 or not top2:
        return 0.0

    intersection = len(top1 & top2)
    union = len(top1 | top2)

    return intersection / union if union > 0 else 0.0

# =============================================================================
# BIDIRECTIONAL EXPANSION (span embedding)
# =============================================================================

def compute_expansion(unit_text: str,
                     current_left: Counter,
                     current_right: Counter,
                     catalog: UnitCatalog,
                     context_threshold: float = 0.30,
                     max_iters: int = 2,
                     max_candidates: int = 200) -> Tuple[Set[str], Set[frozenset]]:
    """
    Compute bidirectional expansion for a span.

    This expansion set IS the span's embedding - it captures which
    units and contexts are substitutable.

    Returns: (unit_expansion, context_expansion)
    """
    # Check cache
    context_hash = hash((frozenset(current_left.items()),
                        frozenset(current_right.items())))
    cache_key = (unit_text, context_hash)
    if cache_key in catalog.expansion_cache:
        return catalog.expansion_cache[cache_key]

    target_unit = catalog.get_unit(unit_text)
    if not target_unit:
        result = (set(), set())
        catalog.expansion_cache[cache_key] = result
        return result

    # Initialize with target unit's typical contexts
    unit_expansion = {unit_text}
    context_expansion = {
        frozenset(target_unit.left_words.most_common(5) +
                 target_unit.right_words.most_common(5))
    }

    # Pre-filter candidates by length for efficiency
    target_length = unit_text.count(' ') + 1
    candidates = [(text, pattern) for text, pattern in catalog.units.items()
                  if (text.count(' ') + 1) == target_length]

    # Limit number of candidates to check
    if len(candidates) > max_candidates:
        import random
        candidates = random.sample(candidates, max_candidates)

    for iteration in range(max_iters):
        # EXPAND UNITS: Find units with similar contexts
        new_units = set()
        for existing_unit in list(unit_expansion):
            existing_pattern = catalog.get_unit(existing_unit)
            if not existing_pattern:
                continue

            for candidate_text, candidate_pattern in candidates:
                if candidate_text in unit_expansion:
                    continue

                # Compute context similarity
                left_sim = context_similarity_aggregated(
                    existing_pattern.left_words,
                    candidate_pattern.left_words
                )
                right_sim = context_similarity_aggregated(
                    existing_pattern.right_words,
                    candidate_pattern.right_words
                )

                avg_sim = (left_sim + right_sim) / 2.0
                if avg_sim >= context_threshold:
                    new_units.add(candidate_text)

                # Limit to prevent explosion
                if len(new_units) > 100:
                    break

            if len(new_units) > 100:
                break

        unit_expansion.update(new_units)

        # EXPAND CONTEXTS: Add typical contexts of new units
        new_context_patterns = set()
        for u in new_units:
            u_pattern = catalog.get_unit(u)
            if u_pattern:
                pattern_repr = frozenset(
                    u_pattern.left_words.most_common(5) +
                    u_pattern.right_words.most_common(5)
                )
                new_context_patterns.add(pattern_repr)

        context_expansion.update(new_context_patterns)

    # Cache result
    result = (unit_expansion, context_expansion)
    catalog.expansion_cache[cache_key] = result
    return result

# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================

def compute_unit_energy(span_state: SpanState,
                       left_context: List[str],
                       right_context: List[str],
                       catalog: UnitCatalog,
                       use_expansion: bool = False) -> float:
    """
    Energy for treating span as a unit.

    Two modes:
    1. use_expansion=True: Based on expansion size (experimental)
    2. use_expansion=False: Based on frequency + compositionality (like prototype)
    """
    unit_pattern = catalog.get_unit(span_state.span.text)

    if unit_pattern is None:
        # Not in catalog - very high energy
        span_state.unit_expansion = set()
        span_state.context_expansion = set()
        return 100.0

    if use_expansion:
        # Expansion-based energy (original Hamiltonian approach)
        current_left = Counter(left_context)
        current_right = Counter(right_context)

        unit_exp, ctx_exp = compute_expansion(
            span_state.span.text, current_left, current_right, catalog
        )

        span_state.unit_expansion = unit_exp
        span_state.context_expansion = ctx_exp

        if len(unit_exp) == 0:
            return 100.0

        combined = len(unit_exp) * math.log(len(ctx_exp) + 1)
        energy = -math.log(combined + 1)
    else:
        # Frequency-based energy (like prototype, more reliable)
        freq_energy = -math.log(unit_pattern.count + 1)

        # Compositionality bonus for multi-word units
        compositionality_bonus = 0.0
        if span_state.span.length >= 2:
            words = span_state.span.tokens
            component_freqs = []
            for w in words:
                w_pattern = catalog.get_unit(w)
                component_freqs.append(w_pattern.count if w_pattern else 1)

            # Expected frequency if components were independent
            expected_freq = 1.0
            for f in component_freqs:
                expected_freq *= (f / 1000000)  # normalize

            if expected_freq > 0:
                ratio = unit_pattern.count / max(expected_freq, 0.01)
                compositionality_bonus = 3.0 * math.log(ratio + 1)

        energy = freq_energy - compositionality_bonus

        # Store minimal expansion info
        span_state.unit_expansion = {span_state.span.text}
        span_state.context_expansion = set()

    return energy

def compute_relational_energy(span1: SpanState,
                              span2: SpanState,
                              relation_type: str) -> float:
    """
    Energy for relational constraint between two spans.

    Relation types:
    - 'adjacent': spans are adjacent in sequence (e.g., left+right children)
    - 'parent-child': parent-child relation
    - 'sibling': sibling relation

    Lower energy when expansions are compatible:
    - Complementary distributions (different but mutually predictive)
    - Joint occurrence patterns from corpus
    """
    # For now, simple compatibility check based on expansion overlap
    # TODO: Implement learned relation types

    if not span1.unit_expansion or not span2.unit_expansion:
        return 0.0

    # Adjacent spans should have complementary, not identical, expansions
    overlap = len(span1.unit_expansion & span2.unit_expansion)
    overlap_ratio = overlap / max(len(span1.unit_expansion),
                                  len(span2.unit_expansion))

    # Prefer some overlap but not complete overlap
    # Complete overlap suggests they should be merged
    # No overlap suggests they're unrelated
    if relation_type == 'adjacent':
        ideal_overlap = 0.3
        deviation = abs(overlap_ratio - ideal_overlap)
        return 2.0 * deviation

    return 0.0

def compute_total_energy(all_spans: List[SpanState],
                        full_tokens: List[str],
                        catalog: UnitCatalog,
                        context_window: int = 3,
                        use_expansion: bool = False) -> float:
    """
    Total Hamiltonian energy for a parse configuration.

    H = sum(unit_energies) + sum(relational_energies) + split_penalties
    """
    total = 0.0

    # Unit energies
    for span_state in all_spans:
        if span_state.is_unit:
            # Extract context
            start = span_state.span.start
            end = span_state.span.end
            left_ctx = full_tokens[max(0, start - context_window):start]
            right_ctx = full_tokens[end:min(len(full_tokens), end + context_window)]

            # Compute unit energy
            energy = compute_unit_energy(span_state, left_ctx, right_ctx, catalog, use_expansion)
            span_state.unit_energy = energy
            total += energy
        else:
            # Split: add child energies (use actual energy, not just unit_energy)
            if span_state.left_child and span_state.right_child:
                total += span_state.left_child.energy
                total += span_state.right_child.energy
                total += 2.0  # Split penalty

                # Relational energy between children
                rel_energy = compute_relational_energy(
                    span_state.left_child,
                    span_state.right_child,
                    'adjacent'
                )
                span_state.relational_energy = rel_energy
                total += rel_energy

    return total

# =============================================================================
# HAMILTONIAN PARSER
# =============================================================================

class HamiltonianParser:
    """
    Parser using Hamiltonian energy minimization.

    Unlike the greedy top-down approach, this jointly optimizes
    all parsing decisions through iterative energy minimization.
    """

    def __init__(self, catalog: UnitCatalog,
                 context_window: int = 3,
                 max_iters: int = 5,
                 use_expansion: bool = False,
                 debug: bool = False):
        self.catalog = catalog
        self.context_window = context_window
        self.max_iters = max_iters
        self.use_expansion = use_expansion
        self.debug = debug

    def parse(self, tokens: List[str]) -> SpanState:
        """
        Parse via iterative energy minimization.

        Algorithm:
        1. Initialize with greedy top-down parse
        2. For each iteration:
           a. Compute current energy
           b. Propose local modifications (change split decisions)
           c. Accept if energy decreases
        3. Return lowest energy configuration
        """
        # Step 1: Initialize with greedy parse
        root_span = Span(0, len(tokens), tokens)
        initial_state = self._greedy_parse(root_span, tokens)

        current_state = initial_state
        current_energy = self._compute_parse_energy(current_state, tokens)

        if self.debug:
            print(f"\nInitial energy: {current_energy:.2f}")

        # Step 2: Iterative refinement
        for iteration in range(self.max_iters):
            # Propose modifications
            proposed_state = self._propose_modification(current_state, tokens)
            proposed_energy = self._compute_parse_energy(proposed_state, tokens)

            # Accept if better
            if proposed_energy < current_energy:
                current_state = proposed_state
                current_energy = proposed_energy

                if self.debug:
                    print(f"Iter {iteration + 1}: energy {current_energy:.2f} (improved)")
            else:
                if self.debug:
                    print(f"Iter {iteration + 1}: energy {current_energy:.2f} (no change)")

        return current_state

    def _greedy_parse(self, span: Span, full_tokens: List[str], depth: int = 0) -> SpanState:
        """Greedy top-down parse (same as bidir_simple)."""
        state = SpanState(span=span)

        # Extract context
        left_ctx = full_tokens[max(0, span.start - self.context_window):span.start]
        right_ctx = full_tokens[span.end:min(len(full_tokens),
                                             span.end + self.context_window)]

        # Test as unit
        unit_energy = compute_unit_energy(state, left_ctx, right_ctx, self.catalog, self.use_expansion)

        if self.debug and span.length >= 2:
            indent = "  " * depth
            print(f"{indent}Testing \"{span.text}\" as unit: E={unit_energy:.2f}")

        # Test splits
        best_split_energy = float('inf')
        best_split = None

        if span.length >= 2:
            for m in range(span.start + 1, span.end):
                left_span = Span(span.start, m, span.tokens[:m - span.start])
                right_span = Span(m, span.end, span.tokens[m - span.start:])

                left_child = self._greedy_parse(left_span, full_tokens, depth + 1)
                right_child = self._greedy_parse(right_span, full_tokens, depth + 1)

                # Use actual energy (unit or split) for each child
                split_energy = (left_child.energy + right_child.energy + 2.0)

                if self.debug and depth == 0:
                    indent = "  " * depth
                    print(f"{indent}  Split at {m}: \"{left_span.text}\" ({left_child.energy:.2f}) | \"{right_span.text}\" ({right_child.energy:.2f}) = {split_energy:.2f}")

                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split = (m, left_child, right_child)

        # Choose unit vs split
        if best_split is None or unit_energy <= best_split_energy:
            state.is_unit = True
            state.unit_energy = unit_energy
            if self.debug and span.length >= 2:
                indent = "  " * depth
                print(f"{indent}→ Chose UNIT for \"{span.text}\": E_unit={unit_energy:.2f} <= E_split={best_split_energy:.2f}")
        else:
            state.is_unit = False
            m, left_child, right_child = best_split
            state.split_point = m
            state.left_child = left_child
            state.right_child = right_child
            state.split_energy = best_split_energy
            if self.debug and span.length >= 2:
                indent = "  " * depth
                left_text = left_child.span.text
                right_text = right_child.span.text
                print(f"{indent}→ Chose SPLIT for \"{span.text}\": \"{left_text}\" | \"{right_text}\", E_split={best_split_energy:.2f} < E_unit={unit_energy:.2f}")

        return state

    def _compute_parse_energy(self, state: SpanState,
                             full_tokens: List[str]) -> float:
        """Compute total energy for a parse tree."""
        # Collect all spans in tree
        all_spans = self._collect_spans(state)
        return compute_total_energy(all_spans, full_tokens,
                                    self.catalog, self.context_window,
                                    self.use_expansion)

    def _collect_spans(self, state: SpanState) -> List[SpanState]:
        """Collect all span states in tree."""
        spans = [state]
        if not state.is_unit and state.left_child and state.right_child:
            spans.extend(self._collect_spans(state.left_child))
            spans.extend(self._collect_spans(state.right_child))
        return spans

    def _propose_modification(self, state: SpanState,
                             full_tokens: List[str]) -> SpanState:
        """
        Propose a modification to the parse tree.

        For now: randomly flip one split decision
        TODO: Smarter proposals based on energy gradients
        """
        import copy
        import random

        # Deep copy current state
        new_state = copy.deepcopy(state)

        # Collect all internal spans (can be modified)
        all_spans = self._collect_spans(new_state)
        modifiable = [s for s in all_spans if s.span.length >= 2]

        if not modifiable:
            return new_state

        # Pick random span to modify
        target = random.choice(modifiable)

        # Flip decision: if unit, try splitting; if split, try merging
        if target.is_unit:
            # Try splitting
            m = random.randint(target.span.start + 1, target.span.end - 1)
            left_span = Span(target.span.start, m,
                           target.span.tokens[:m - target.span.start])
            right_span = Span(m, target.span.end,
                            target.span.tokens[m - target.span.start:])

            target.is_unit = False
            target.split_point = m
            target.left_child = self._greedy_parse(left_span, full_tokens)
            target.right_child = self._greedy_parse(right_span, full_tokens)
        else:
            # Try merging (treat as unit)
            target.is_unit = True
            target.split_point = None
            target.left_child = None
            target.right_child = None

            # Recompute unit energy
            left_ctx = full_tokens[max(0, target.span.start - self.context_window):
                                  target.span.start]
            right_ctx = full_tokens[target.span.end:
                                   min(len(full_tokens),
                                       target.span.end + self.context_window)]
            target.unit_energy = compute_unit_energy(target, left_ctx,
                                                     right_ctx, self.catalog,
                                                     self.use_expansion)

        return new_state

# =============================================================================
# VISUALIZATION
# =============================================================================

def print_parse_tree(state: SpanState, indent="", is_last=True):
    """Pretty print parse tree."""
    branch = "└─" if is_last else "├─"
    status = "UNIT" if state.is_unit else "SPLIT"

    info = f"E={state.unit_energy:.2f}" if state.is_unit else f"E={state.split_energy:.2f}"
    if state.is_unit and state.unit_expansion:
        info += f" U={len(state.unit_expansion)} C={len(state.context_expansion)}"

    print(f"{indent}{branch} [{status}] {info} \"{state.span.text}\"")

    if not state.is_unit and state.left_child and state.right_child:
        child_indent = indent + ("   " if is_last else "│  ")
        print_parse_tree(state.left_child, child_indent, False)
        print_parse_tree(state.right_child, child_indent, True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=str, default="unit_catalog.pkl")
    ap.add_argument("--sentence", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max-iters", type=int, default=5)
    args = ap.parse_args()

    print("=" * 80)
    print("HAMILTONIAN DYNAMIC PARSER")
    print("Iterative Energy Minimization with Relational Constraints")
    print("=" * 80)

    # Load catalog
    catalog = UnitCatalog()
    try:
        catalog.load(args.catalog)
    except FileNotFoundError:
        print(f"Error: Catalog {args.catalog} not found")
        print("Run bidir_simple.py first to build unit_catalog.pkl")
        return

    print("\n" + "=" * 80)
    print("PARSING")
    print("=" * 80)

    # Test sentences
    if args.sentence:
        test_sentences = [args.sentence]
    else:
        test_sentences = [
            "did you go out with sarah",
            "should we go out on friday",
            "i go out the door",
            "i lost my money yesterday",
            "my money is gone",
        ]

    parser = HamiltonianParser(catalog, debug=args.debug, max_iters=args.max_iters)

    for sent in test_sentences:
        print(f"\n{'-' * 80}")
        print(f"SENTENCE: \"{sent}\"")
        print('-' * 80)

        tokens = sent.lower().split()
        parse_tree = parser.parse(tokens)

        print("\nFINAL PARSE TREE:")
        print_parse_tree(parse_tree)

if __name__ == "__main__":
    main()
