#!/usr/bin/env python3
"""
PROTOTYPE: Top-Down Span Parsing with Unit Catalog

Core ideas to test:
1. Build catalog of multi-word units from corpus
2. Parse top-down: test spans as units BEFORE splitting
3. Score units by corpus frequency + context match
4. Compare unit energy vs split energy

This is a minimal prototype - no iterative minimization yet, just greedy decisions.
"""

import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContextPattern:
    """Stores context patterns for a unit."""
    left_words: Counter  # words appearing to the left
    right_words: Counter  # words appearing to the right
    count: int  # how many times this unit appears

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

@dataclass
class ParseNode:
    """Node in parse tree."""
    span: Span
    energy: float
    split_point: Optional[int] = None
    left: Optional["ParseNode"] = None
    right: Optional["ParseNode"] = None
    unit_score: float = 0.0

    def is_leaf(self):
        return self.left is None and self.right is None

# =============================================================================
# UNIT CATALOG BUILDER
# =============================================================================

class UnitCatalog:
    """
    Catalog of n-gram units (n=1,2,3,4) with their corpus contexts.
    """

    def __init__(self, min_freq=5, max_ngram=4, context_window=3):
        self.min_freq = min_freq
        self.max_ngram = max_ngram
        self.context_window = context_window
        self.units = {}  # text -> ContextPattern

    def build_from_corpus(self, corpus_file: str):
        """Extract all n-grams and their contexts from corpus."""
        print(f"Building unit catalog from {corpus_file}...")

        # Read corpus
        with open(corpus_file, 'r', encoding='utf-8') as f:
            text = f.read().lower()

        tokens = text.split()
        print(f"Total tokens: {len(tokens):,}")

        # Extract all n-grams
        ngram_contexts = defaultdict(lambda: {
            'count': 0,
            'left_words': Counter(),
            'right_words': Counter()
        })

        for n in range(1, self.max_ngram + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])

                # Extract context
                left_start = max(0, i - self.context_window)
                left_context = tokens[left_start:i]

                right_end = min(len(tokens), i + n + self.context_window)
                right_context = tokens[i+n:right_end]

                # Store
                ngram_contexts[ngram]['count'] += 1
                ngram_contexts[ngram]['left_words'].update(left_context)
                ngram_contexts[ngram]['right_words'].update(right_context)

        # Filter by frequency and store
        for ngram, data in ngram_contexts.items():
            if data['count'] >= self.min_freq:
                self.units[ngram] = ContextPattern(
                    left_words=data['left_words'],
                    right_words=data['right_words'],
                    count=data['count']
                )

        print(f"Extracted {len(self.units):,} units (min_freq={self.min_freq})")
        print(f"  1-grams: {sum(1 for u in self.units if ' ' not in u):,}")
        print(f"  2-grams: {sum(1 for u in self.units if u.count(' ') == 1):,}")
        print(f"  3-grams: {sum(1 for u in self.units if u.count(' ') == 2):,}")
        print(f"  4-grams: {sum(1 for u in self.units if u.count(' ') == 3):,}")

    def get_unit(self, text: str) -> Optional[ContextPattern]:
        """Look up a unit in the catalog."""
        return self.units.get(text)

    def save(self, path: str):
        """Save catalog to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.units, f)
        print(f"Saved catalog to {path}")

    def load(self, path: str):
        """Load catalog from file."""
        import pickle
        with open(path, 'rb') as f:
            self.units = pickle.load(f)
        print(f"Loaded {len(self.units):,} units from {path}")

# =============================================================================
# ENERGY FUNCTIONS
# =============================================================================

def compute_unit_energy(span: Span, catalog: UnitCatalog,
                        left_context: List[str], right_context: List[str]) -> float:
    """
    Energy for treating span as a unit.
    Lower energy = better unit.

    E_unit = -log(freq) + context_mismatch - compositionality_bonus
    """
    unit = catalog.get_unit(span.text)

    if unit is None:
        # Not in catalog - very high energy
        return 100.0

    # Base energy from frequency (higher freq = lower energy)
    freq_energy = -math.log(unit.count + 1)

    # Context match energy (lower = better match)
    context_mismatch = compute_context_mismatch(
        left_context, right_context, unit
    )

    # COMPOSITIONALITY BONUS: Multi-word units get bonus if they're actually compositional
    # If a 2-word unit exists in corpus, it's likely meaningful
    compositionality_bonus = 0.0
    if span.length >= 2:
        # Bonus proportional to: (multi-word freq) / (product of component freqs)
        # High ratio = unit appears together more than expected
        words = span.tokens
        component_freqs = [catalog.get_unit(w).count if catalog.get_unit(w) else 1
                          for w in words]
        expected_freq = 1
        for f in component_freqs:
            expected_freq *= (f / 1000000)  # normalize

        if expected_freq > 0:
            ratio = unit.count / max(expected_freq, 0.01)
            compositionality_bonus = 3.0 * math.log(ratio + 1)  # Strong bonus for non-compositional units

    total_energy = freq_energy + 5.0 * context_mismatch - compositionality_bonus
    return total_energy

def compute_context_mismatch(left_context: List[str], right_context: List[str],
                              unit: ContextPattern) -> float:
    """
    How well does current context match corpus contexts for this unit?
    Returns 0 if perfect match, higher values for mismatch.
    """
    # Convert current context to counter
    left_current = Counter(left_context)
    right_current = Counter(right_context)

    # Compute overlap (simple Jaccard-like measure)
    def overlap_score(current: Counter, corpus: Counter) -> float:
        if not current or not corpus:
            return 0.5

        # Words in common
        common = set(current.keys()) & set(corpus.keys())
        union = set(current.keys()) | set(corpus.keys())

        if not union:
            return 0.5

        return 1.0 - (len(common) / len(union))

    left_mismatch = overlap_score(left_current, unit.left_words)
    right_mismatch = overlap_score(right_current, unit.right_words)

    return (left_mismatch + right_mismatch) / 2.0

def compute_split_energy(left_child: ParseNode, right_child: ParseNode) -> float:
    """
    Energy for splitting into two children.
    Simple version: sum of child energies with penalty.
    """
    split_penalty = 2.0  # penalty for creating a split (increased to favor units)
    return left_child.energy + right_child.energy + split_penalty

# =============================================================================
# TOP-DOWN PARSER
# =============================================================================

class TopDownUnitParser:
    """
    Parses by testing spans as units before splitting.
    Greedy version - no iterative energy minimization yet.
    """

    def __init__(self, catalog: UnitCatalog, min_span_len=1, context_window=3):
        self.catalog = catalog
        self.min_span_len = min_span_len
        self.context_window = context_window

    def parse(self, tokens: List[str]) -> ParseNode:
        """Parse a list of tokens top-down."""
        span = Span(0, len(tokens), tokens)
        return self._parse_span(span, tokens)

    def _parse_span(self, span: Span, full_tokens: List[str]) -> ParseNode:
        """
        Recursively parse a span.
        1. Test as unit
        2. Test all splits
        3. Choose lowest energy
        """
        # Extract context for this span
        left_context = self._get_left_context(span.start, full_tokens)
        right_context = self._get_right_context(span.end, full_tokens)

        # 1. TEST AS UNIT
        unit_energy = compute_unit_energy(span, self.catalog, left_context, right_context)

        # DEBUG: Show unit testing for multi-word spans
        if span.length >= 2 and span.length <= 3:
            unit = self.catalog.get_unit(span.text)
            if unit:
                print(f"  DEBUG: Testing \"{span.text}\" as unit: E={unit_energy:.2f}, freq={unit.count}")
            else:
                print(f"  DEBUG: Testing \"{span.text}\" as unit: NOT IN CATALOG (E={unit_energy:.2f})")

        # 2. TEST SPLITS (if span is long enough)
        best_split_energy = float('inf')
        best_split = None

        if span.length >= 2 * self.min_span_len:
            for m in range(span.start + self.min_span_len,
                          span.end - self.min_span_len + 1):
                # Create child spans
                left_span = Span(span.start, m, span.tokens[:m-span.start])
                right_span = Span(m, span.end, span.tokens[m-span.start:])

                # Recursively parse children
                left_child = self._parse_span(left_span, full_tokens)
                right_child = self._parse_span(right_span, full_tokens)

                # Compute split energy
                split_energy = compute_split_energy(left_child, right_child)

                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split = (m, left_child, right_child)

        # 3. CHOOSE: unit vs best split
        if best_split is None or unit_energy <= best_split_energy:
            # Keep as unit (leaf node)
            unit = self.catalog.get_unit(span.text)
            unit_score = unit.count if unit else 0
            return ParseNode(span, unit_energy, unit_score=unit_score)
        else:
            # Split (internal node)
            m, left_child, right_child = best_split
            return ParseNode(span, best_split_energy,
                           split_point=m, left=left_child, right=right_child)

    def _get_left_context(self, pos: int, tokens: List[str]) -> List[str]:
        """Get left context words."""
        start = max(0, pos - self.context_window)
        return tokens[start:pos]

    def _get_right_context(self, pos: int, tokens: List[str]) -> List[str]:
        """Get right context words."""
        end = min(len(tokens), pos + self.context_window)
        return tokens[pos:end]

# =============================================================================
# VISUALIZATION
# =============================================================================

def print_tree(node: ParseNode, indent="", is_last=True):
    """Pretty print parse tree."""
    branch = "└─" if is_last else "├─"

    status = "UNIT" if node.is_leaf() else "SPLIT"
    info = f"E={node.energy:.2f}"
    if node.is_leaf() and node.unit_score > 0:
        info += f" freq={node.unit_score}"

    print(f"{indent}{branch} [{status}] {info} \"{node.span.text}\"")

    if not node.is_leaf():
        child_indent = indent + ("   " if is_last else "│  ")
        print_tree(node.left, child_indent, False)
        print_tree(node.right, child_indent, True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="dialog_corpus.txt",
                    help="Corpus file to build catalog from")
    ap.add_argument("--build-catalog", action="store_true",
                    help="Build and save catalog")
    ap.add_argument("--catalog-path", type=str, default="unit_catalog.pkl",
                    help="Path to save/load catalog")
    ap.add_argument("--sentence", type=str, default=None,
                    help="Sentence to parse (if not provided, will use examples)")
    ap.add_argument("--min-freq", type=int, default=10,
                    help="Minimum frequency for units")
    args = ap.parse_args()

    print("=" * 80)
    print("PROTOTYPE: TOP-DOWN SPAN PARSING WITH UNIT CATALOG")
    print("=" * 80)

    # Build or load catalog
    catalog = UnitCatalog(min_freq=args.min_freq)

    if args.build_catalog:
        catalog.build_from_corpus(args.corpus)
        catalog.save(args.catalog_path)
    else:
        try:
            catalog.load(args.catalog_path)
        except FileNotFoundError:
            print(f"Catalog not found at {args.catalog_path}")
            print("Building from corpus...")
            catalog.build_from_corpus(args.corpus)
            catalog.save(args.catalog_path)

    print("\n" + "=" * 80)
    print("PARSING EXAMPLES")
    print("=" * 80)

    # Test sentences
    if args.sentence:
        test_sentences = [args.sentence]
    else:
        test_sentences = [
            "did you go out with sarah",
            "i go out the door",
            "i lost my money yesterday",
            "i lost my mind completely",
            "go out together tonight",
            "my money is gone",
            "the bank by the river",
            "i went to the bank",
        ]

    parser = TopDownUnitParser(catalog)

    for sent in test_sentences:
        print(f"\n{'-' * 80}")
        print(f"SENTENCE: \"{sent}\"")
        print('-' * 80)

        tokens = sent.lower().split()
        tree = parser.parse(tokens)

        print("\nPARSE TREE:")
        print_tree(tree)

        # Show what units were found
        def collect_units(node):
            if node.is_leaf() and node.unit_score > 0:
                return [(node.span.text, node.unit_score)]
            elif not node.is_leaf():
                return collect_units(node.left) + collect_units(node.right)
            return []

        units = collect_units(tree)
        if units:
            print("\nUNITS FOUND (with corpus frequency):")
            for text, freq in units:
                print(f"  - \"{text}\" (freq={freq})")

if __name__ == "__main__":
    main()
