#!/usr/bin/env python3
"""
Simplified Bidirectional Expansion

Demonstrates the concept without storing all context instances.
Uses aggregated context patterns (Counter objects) from the prototype.
"""

#!/usr/bin/env python3
from collections import Counter
from typing import Set, Tuple
import math
import pickle

# Import/define data structures from prototype
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ContextPattern:
    """Stores context patterns for a unit."""
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

@dataclass
class ParseNode:
    """Node in parse tree."""
    span: Span
    energy: float
    unit_expansion: int = 0
    context_expansion: int = 0
    split_point: Optional[int] = None
    left: Optional["ParseNode"] = None
    right: Optional["ParseNode"] = None

    def is_leaf(self):
        return self.left is None and self.right is None

def print_tree(node: ParseNode, indent="", is_last=True):
    """Pretty print parse tree."""
    branch = "└─" if is_last else "├─"
    status = "UNIT" if node.is_leaf() else "SPLIT"
    info = f"E={node.energy:.2f}"
    if node.is_leaf() and node.unit_expansion > 0:
        info += f" U={node.unit_expansion} C={node.context_expansion}"
    print(f"{indent}{branch} [{status}] {info} \"{node.span.text}\"")
    if not node.is_leaf():
        child_indent = indent + ("   " if is_last else "│  ")
        print_tree(node.left, child_indent, False)
        print_tree(node.right, child_indent, True)

class UnitCatalog:
    """Simple catalog loader."""
    def __init__(self):
        self.units = {}

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.units = pickle.load(f)
        print(f"Loaded {len(self.units):,} units from {path}")

    def get_unit(self, text: str) -> Optional[ContextPattern]:
        return self.units.get(text)

def context_similarity_aggregated(ctx1_counter: Counter, ctx2_counter: Counter) -> float:
    """
    Similarity between two aggregated context distributions.
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

def bidirectional_expansion_simple(unit_text: str,
                                  current_left: Counter,
                                  current_right: Counter,
                                  catalog: UnitCatalog,
                                  context_threshold: float = 0.30,
                                  max_iters: int = 2) -> Tuple[Set[str], Set[str]]:
    """
    Bidirectional expansion using aggregated contexts.

    Returns: (unit_expansion, context_pattern_expansion)
    """
    target_unit = catalog.get_unit(unit_text)
    if not target_unit:
        return (set(), set())

    # Initialize - start with target unit's typical contexts
    unit_expansion = {unit_text}
    context_expansion = {frozenset(target_unit.left_words.most_common(5) +
                                   target_unit.right_words.most_common(5))}

    for iteration in range(max_iters):
        # EXPAND UNITS: Find units with similar aggregated contexts to ANY unit in expansion
        new_units = set()
        for existing_unit in list(unit_expansion):
            existing_pattern = catalog.get_unit(existing_unit)
            if not existing_pattern:
                continue

            for candidate_text, candidate_pattern in catalog.units.items():
                if candidate_text in unit_expansion:
                    continue

                # Skip if different length
                candidate_len = candidate_text.count(' ') + 1  # number of words
                existing_len = existing_unit.count(' ') + 1
                if candidate_len != existing_len:
                    continue

                # Compute similarity to existing unit's contexts
                left_sim = context_similarity_aggregated(existing_pattern.left_words,
                                                        candidate_pattern.left_words)
                right_sim = context_similarity_aggregated(existing_pattern.right_words,
                                                         candidate_pattern.right_words)

                avg_sim = (left_sim + right_sim) / 2.0
                if avg_sim >= context_threshold:
                    new_units.add(candidate_text)

            # Limit expansion to prevent explosion
            if len(new_units) > 100:
                break

        unit_expansion.update(new_units)

        # EXPAND CONTEXTS: Add typical contexts of new units
        new_context_patterns = set()
        for u in new_units:
            u_pattern = catalog.get_unit(u)
            if u_pattern:
                pattern_repr = frozenset(u_pattern.left_words.most_common(5) +
                                        u_pattern.right_words.most_common(5))
                new_context_patterns.add(pattern_repr)

        context_expansion.update(new_context_patterns)

    return (unit_expansion, context_expansion)

def compute_bidir_energy(span: Span,
                         left_context,
                         right_context,
                         catalog: UnitCatalog,
                         debug=False) -> Tuple[float, int, int]:
    """Energy via simplified bidirectional expansion."""
    current_left = Counter(left_context)
    current_right = Counter(right_context)

    unit_exp, ctx_exp = bidirectional_expansion_simple(
        span.text, current_left, current_right, catalog
    )

    unit_exp_size = len(unit_exp)
    ctx_exp_size = len(ctx_exp)

    if unit_exp_size == 0:
        return (100.0, 0, 0)

    # Energy from combined expansion
    combined = unit_exp_size * math.log(ctx_exp_size + 1)
    energy = -math.log(combined + 1)

    if debug and span.length >= 2:
        print(f"  \"{span.text}\": U={unit_exp_size}, C={ctx_exp_size}, E={energy:.2f}")
        if unit_exp_size <= 10:
            print(f"    expansion: {list(unit_exp)[:10]}")

    return (energy, unit_exp_size, ctx_exp_size)

class SimpleBidirParser:
    """Parser using simplified bidirectional expansion."""

    def __init__(self, catalog, context_window=3, debug=False):
        self.catalog = catalog
        self.context_window = context_window
        self.debug = debug

    def parse(self, tokens):
        span = Span(0, len(tokens), tokens)
        return self._parse_span(span, tokens)

    def _parse_span(self, span, full_tokens):
        left_ctx = full_tokens[max(0, span.start - self.context_window):span.start]
        right_ctx = full_tokens[span.end:min(len(full_tokens), span.end + self.context_window)]

        # Test as unit
        unit_energy, u_exp, c_exp = compute_bidir_energy(
            span, left_ctx, right_ctx, self.catalog, self.debug
        )

        # Test splits
        best_split_energy = float('inf')
        best_split = None

        if span.length >= 2:
            for m in range(span.start + 1, span.end):
                left_span = Span(span.start, m, span.tokens[:m-span.start])
                right_span = Span(m, span.end, span.tokens[m-span.start:])

                left_child = self._parse_span(left_span, full_tokens)
                right_child = self._parse_span(right_span, full_tokens)

                split_energy = left_child.energy + right_child.energy + 2.0

                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split = (m, left_child, right_child)

        # Choose
        if best_split is None or unit_energy <= best_split_energy:
            node = ParseNode(span, unit_energy)
            node.unit_expansion = u_exp
            node.context_expansion = c_exp
            return node
        else:
            m, left_child, right_child = best_split
            node = ParseNode(span, best_split_energy, split_point=m,
                           left=left_child, right=right_child)
            return node

def main():
    print("=" * 80)
    print("SIMPLIFIED BIDIRECTIONAL EXPANSION PARSER")
    print("=" * 80)

    # Load existing catalog
    catalog = UnitCatalog()
    try:
        catalog.load('unit_catalog.pkl')
    except:
        print("Error: Run prototype first to build unit_catalog.pkl")
        return

    print("\n" + "=" * 80)
    print("PARSING")
    print("=" * 80)

    test_sentences = [
        "did you go out with sarah",
        "should we go out on friday",  # Novel context!
        "i go out the door",
        "i lost my money yesterday",
        "my money is gone",
    ]

    parser = SimpleBidirParser(catalog, debug=True)

    for sent in test_sentences:
        print(f"\n{'-' * 80}")
        print(f"SENTENCE: \"{sent}\"")
        print('-' * 80)

        tokens = sent.lower().split()
        tree = parser.parse(tokens)

        print("\nPARSE TREE:")
        print_tree(tree)

if __name__ == "__main__":
    main()
