#!/usr/bin/env python3
"""
Simplified Bidirectional Expansion

Demonstrates the concept without storing all context instances.
Uses aggregated context patterns (Counter objects) from the prototype.
"""

from collections import Counter
from typing import Set, Tuple, List
import math
import pickle
import numpy as np

# PyTorch for GPU acceleration (MPS on Apple Silicon)
try:
    import torch
    HAS_TORCH = True
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = None

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
    """Catalog with optional GPU-accelerated similarity computation."""

    def __init__(self):
        self.units = {}
        self.gpu_ready = False
        # GPU tensors (built on demand)
        self.gpu_left = None
        self.gpu_right = None
        self.unit_list = None
        self.unit_to_idx = None
        # Length index for fast filtering
        self.units_by_length = {}

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.units = pickle.load(f)
        print(f"Loaded {len(self.units):,} units from {path}")
        # Build length index
        self._build_length_index()

    def _build_length_index(self):
        """Index units by word count for fast length filtering."""
        self.units_by_length = {}
        for text in self.units:
            length = text.count(' ') + 1
            if length not in self.units_by_length:
                self.units_by_length[length] = []
            self.units_by_length[length].append(text)

    def get_unit(self, text: str) -> Optional[ContextPattern]:
        return self.units.get(text)

    def build_gpu_index(self, min_freq: int = 10):
        """Build GPU tensors for fast batch similarity computation."""
        if not HAS_TORCH or DEVICE is None:
            print("GPU not available, using CPU fallback")
            return

        print(f"Building GPU index on {DEVICE}...")

        # Filter to frequent units
        freq_units = {text: pattern for text, pattern in self.units.items()
                      if pattern.count >= min_freq}

        # Build unit list and index
        self.unit_list = list(freq_units.keys())
        self.unit_to_idx = {text: i for i, text in enumerate(self.unit_list)}
        n_units = len(self.unit_list)

        # Build vocabulary from top-k context words
        all_words = set()
        for pattern in freq_units.values():
            all_words.update(w for w, _ in pattern.left_words.most_common(10))
            all_words.update(w for w, _ in pattern.right_words.most_common(10))

        vocab = sorted(all_words)
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        vocab_size = len(vocab)

        print(f"  {n_units:,} units, {vocab_size:,} vocabulary")

        # Build dense matrices (float16 for memory efficiency)
        cpu_left = np.zeros((n_units, vocab_size), dtype=np.float16)
        cpu_right = np.zeros((n_units, vocab_size), dtype=np.float16)

        for i, text in enumerate(self.unit_list):
            pattern = freq_units[text]

            for w, c in pattern.left_words.most_common(10):
                if w in word_to_idx:
                    cpu_left[i, word_to_idx[w]] = c

            for w, c in pattern.right_words.most_common(10):
                if w in word_to_idx:
                    cpu_right[i, word_to_idx[w]] = c

        # Normalize
        left_norms = np.linalg.norm(cpu_left.astype(np.float32), axis=1, keepdims=True) + 1e-10
        right_norms = np.linalg.norm(cpu_right.astype(np.float32), axis=1, keepdims=True) + 1e-10
        cpu_left = (cpu_left / left_norms).astype(np.float16)
        cpu_right = (cpu_right / right_norms).astype(np.float16)

        # Transfer to GPU
        self.gpu_left = torch.from_numpy(cpu_left).to(DEVICE)
        self.gpu_right = torch.from_numpy(cpu_right).to(DEVICE)

        self.gpu_ready = True
        mem_mb = (self.gpu_left.numel() + self.gpu_right.numel()) * 2 / (1024 * 1024)
        print(f"  GPU index ready ({mem_mb:.1f} MB)")

    def gpu_find_similar(self, unit_text: str, threshold: float = 0.3,
                         same_length: bool = True, max_results: int = 100,
                         batch_size: int = 10000) -> List[Tuple[str, float]]:
        """
        Find units with similar context patterns using GPU.
        Uses batching to avoid memory issues with large candidate sets.

        Args:
            unit_text: Query unit
            threshold: Minimum similarity
            same_length: Only return units with same word count
            max_results: Maximum results to return
            batch_size: Process candidates in batches of this size

        Returns: List of (unit_text, similarity) tuples
        """
        if not self.gpu_ready:
            return self._cpu_find_similar(unit_text, threshold, same_length, max_results)

        if unit_text not in self.unit_to_idx:
            return []

        query_idx = self.unit_to_idx[unit_text]
        query_length = unit_text.count(' ') + 1

        # Get candidates (optionally filter by length)
        if same_length and query_length in self.units_by_length:
            candidates = [t for t in self.units_by_length[query_length]
                          if t in self.unit_to_idx and t != unit_text]
        else:
            candidates = [t for t in self.unit_list if t != unit_text]

        if not candidates:
            return []

        # Get query vectors
        query_left = self.gpu_left[query_idx]
        query_right = self.gpu_right[query_idx]

        # Process in batches to avoid memory issues
        all_results = []
        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]

            cand_indices = torch.tensor([self.unit_to_idx[t] for t in batch_candidates],
                                        dtype=torch.long, device=DEVICE)

            # Gather candidate vectors
            cand_left = self.gpu_left[cand_indices]
            cand_right = self.gpu_right[cand_indices]

            # Cosine similarity (vectors are normalized)
            left_sim = torch.mv(cand_left.float(), query_left.float())
            right_sim = torch.mv(cand_right.float(), query_right.float())
            avg_sim = (left_sim + right_sim) / 2.0

            # Filter by threshold within this batch
            mask = avg_sim >= threshold
            batch_matching_indices = torch.where(mask)[0]
            batch_matching_sims = avg_sim[mask]

            for i, sim in zip(batch_matching_indices.cpu().tolist(),
                              batch_matching_sims.cpu().tolist()):
                all_results.append((batch_candidates[i], sim))

        # Sort all results and return top
        all_results.sort(key=lambda x: -x[1])
        return all_results[:max_results]

    def _cpu_find_similar(self, unit_text: str, threshold: float,
                          same_length: bool, max_results: int) -> List[Tuple[str, float]]:
        """CPU fallback for similarity search."""
        pattern = self.get_unit(unit_text)
        if not pattern:
            return []

        query_length = unit_text.count(' ') + 1
        results = []

        candidates = self.units_by_length.get(query_length, []) if same_length else self.units.keys()

        for cand_text in candidates:
            if cand_text == unit_text:
                continue
            cand_pattern = self.units.get(cand_text)
            if not cand_pattern:
                continue

            left_sim = context_similarity_aggregated(pattern.left_words, cand_pattern.left_words)
            right_sim = context_similarity_aggregated(pattern.right_words, cand_pattern.right_words)
            avg_sim = (left_sim + right_sim) / 2.0

            if avg_sim >= threshold:
                results.append((cand_text, avg_sim))

        results.sort(key=lambda x: -x[1])
        return results[:max_results]

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
    GPU-accelerated when catalog.gpu_ready is True.

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
            if existing_unit not in catalog.units:
                continue

            # GPU-ACCELERATED: Use catalog.gpu_find_similar for batch similarity
            similar_units = catalog.gpu_find_similar(
                existing_unit,
                threshold=context_threshold,
                same_length=True,  # Only same-length units
                max_results=100
            )

            for cand_text, sim in similar_units:
                if cand_text not in unit_expansion:
                    new_units.add(cand_text)

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
