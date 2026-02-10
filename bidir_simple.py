#!/usr/bin/env python3
"""
Simplified Bidirectional Expansion

Demonstrates the concept without storing all context instances.
Uses aggregated context patterns (Counter objects) from the prototype.
"""

from collections import Counter
from typing import Set, Tuple, List, Dict, Optional
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
    unit_expansion_set: Optional[Set[str]] = None  # Actual substitution class
    context_expansion_set: Optional[Set] = None  # Actual context patterns
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
        # Reverse lookup indexes: context word -> set of units
        self.left_word_to_units = {}   # word -> {(unit_text, count), ...}
        self.right_word_to_units = {}  # word -> {(unit_text, count), ...}

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.units = pickle.load(f)
        print(f"Loaded {len(self.units):,} units from {path}")
        # Build length index
        self._build_length_index()
        # Build reverse lookup indexes
        self._build_reverse_indexes()

    def _build_length_index(self):
        """Index units by word count for fast length filtering."""
        self.units_by_length = {}
        for text in self.units:
            length = text.count(' ') + 1
            if length not in self.units_by_length:
                self.units_by_length[length] = []
            self.units_by_length[length].append(text)

    def _build_reverse_indexes(self):
        """Build reverse lookup: context word -> units that have it."""
        from collections import defaultdict

        self.left_word_to_units = defaultdict(list)
        self.right_word_to_units = defaultdict(list)

        for unit_text, pattern in self.units.items():
            # Index left context words
            for word, count in pattern.left_words.items():
                self.left_word_to_units[word].append((unit_text, count))
            # Index right context words
            for word, count in pattern.right_words.items():
                self.right_word_to_units[word].append((unit_text, count))

        # Convert to regular dicts for pickle compatibility
        self.left_word_to_units = dict(self.left_word_to_units)
        self.right_word_to_units = dict(self.right_word_to_units)

        print(f"  Built reverse indexes: {len(self.left_word_to_units):,} left words, "
              f"{len(self.right_word_to_units):,} right words")

    def reverse_left_lookup(self, word: str, min_count: int = 1) -> Set[str]:
        """Find units that have 'word' in their left context."""
        if word not in self.left_word_to_units:
            return set()
        return {unit_text for unit_text, count in self.left_word_to_units[word]
                if count >= min_count}

    def reverse_right_lookup(self, word: str, min_count: int = 1) -> Set[str]:
        """Find units that have 'word' in their right context."""
        if word not in self.right_word_to_units:
            return set()
        return {unit_text for unit_text, count in self.right_word_to_units[word]
                if count >= min_count}

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

        self.vocab = sorted(all_words)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        vocab_size = len(self.vocab)

        print(f"  {n_units:,} units, {vocab_size:,} vocabulary")

        # Build dense matrices (float16 for memory efficiency)
        cpu_left = np.zeros((n_units, vocab_size), dtype=np.float16)
        cpu_right = np.zeros((n_units, vocab_size), dtype=np.float16)

        for i, text in enumerate(self.unit_list):
            pattern = freq_units[text]

            for w, c in pattern.left_words.most_common(10):
                if w in self.word_to_idx:
                    cpu_left[i, self.word_to_idx[w]] = c

            for w, c in pattern.right_words.most_common(10):
                if w in self.word_to_idx:
                    cpu_right[i, self.word_to_idx[w]] = c

        # Normalize
        left_norms = np.linalg.norm(cpu_left.astype(np.float32), axis=1, keepdims=True) + 1e-10
        right_norms = np.linalg.norm(cpu_right.astype(np.float32), axis=1, keepdims=True) + 1e-10
        cpu_left = (cpu_left / left_norms).astype(np.float16)
        cpu_right = (cpu_right / right_norms).astype(np.float16)

        #rjf edit to reduce unneeded digits to reduce memory footprint
        cpu_left = np.round(cpu_left / left_norms, decimals=3).astype(np.float16)
        cpu_right = np.round(cpu_right / right_norms, decimals=3).astype(np.float16)

        # Transfer to GPU
        self.gpu_left = torch.from_numpy(cpu_left).to(DEVICE)
        self.gpu_right = torch.from_numpy(cpu_right).to(DEVICE)

        # Compute information content weights for each vocabulary word
        # IC(word) = -log(P(word appears in context)) = -log(count / n_units)
        self._compute_ic_weights(n_units)

        self.gpu_ready = True
        mem_mb = (self.gpu_left.numel() + self.gpu_right.numel()) * 2 / (1024 * 1024)
        print(f"  GPU index ready ({mem_mb:.1f} MB)")

        # Build index-based reverse lookups and length index for GPU operations
        self._build_gpu_reverse_indexes()

    def _compute_ic_weights(self, n_units: int):
        """
        Compute information content weights for ALL context words.

        IC(word) = -log(P(word appears in context position))
                 = -log(units_with_word / n_units)

        Rare context words have high IC (more informative).
        Common context words have low IC (less informative).
        """
        from collections import Counter

        # Count how many units have each word in left/right context (ALL words)
        left_unit_counts = Counter()  # word -> number of units that have it in left context
        right_unit_counts = Counter()

        for text in self.unit_list:
            pattern = self.units.get(text)
            if pattern:
                # Count each word once per unit (presence, not frequency)
                for word in pattern.left_words:
                    left_unit_counts[word] += 1
                for word in pattern.right_words:
                    right_unit_counts[word] += 1

        # Compute IC = -log(count / n_units) with smoothing
        # Higher IC = rarer = more informative
        eps = 1.0
        self.ic_left_dict = {}
        self.ic_right_dict = {}

        for word, count in left_unit_counts.items():
            self.ic_left_dict[word] = -math.log((count + eps) / (n_units + eps))

        for word, count in right_unit_counts.items():
            self.ic_right_dict[word] = -math.log((count + eps) / (n_units + eps))

        # Also build GPU tensors for IC weights (for IC-weighted cosine scoring)
        # These are indexed by vocabulary position (matching gpu_left/gpu_right)
        vocab_size = len(self.vocab)
        ic_left_array = np.zeros(vocab_size, dtype=np.float32)
        ic_right_array = np.zeros(vocab_size, dtype=np.float32)

        for word, idx in self.word_to_idx.items():
            if word in self.ic_left_dict:
                ic_left_array[idx] = self.ic_left_dict[word]
            if word in self.ic_right_dict:
                ic_right_array[idx] = self.ic_right_dict[word]

        self.gpu_ic_left = torch.from_numpy(ic_left_array).to(DEVICE)
        self.gpu_ic_right = torch.from_numpy(ic_right_array).to(DEVICE)

        max_left = max(self.ic_left_dict.values()) if self.ic_left_dict else 0
        max_right = max(self.ic_right_dict.values()) if self.ic_right_dict else 0
        print(f"  IC weights computed for {len(self.ic_left_dict):,} left words, {len(self.ic_right_dict):,} right words")
        print(f"    (left max={max_left:.2f}, right max={max_right:.2f})")

    def _build_gpu_reverse_indexes(self):
        """Build reverse lookups that return indices (for GPU gathering)."""
        from collections import defaultdict

        # Index by length (using indices)
        self.idx_by_length = defaultdict(list)
        for idx, text in enumerate(self.unit_list):
            length = text.count(' ') + 1
            self.idx_by_length[length].append(idx)

        # Reverse lookups: word -> set of unit indices
        self.left_word_to_idx = defaultdict(set)
        self.right_word_to_idx = defaultdict(set)

        for idx, text in enumerate(self.unit_list):
            pattern = self.units.get(text)
            if pattern:
                for word in pattern.left_words:
                    self.left_word_to_idx[word].add(idx)
                for word in pattern.right_words:
                    self.right_word_to_idx[word].add(idx)

        # Convert to regular dicts
        self.idx_by_length = dict(self.idx_by_length)
        self.left_word_to_idx = dict(self.left_word_to_idx)
        self.right_word_to_idx = dict(self.right_word_to_idx)

        print(f"  GPU reverse indexes ready")

    def gpu_contextual_candidates(self, unit_text: str,
                               left_context: List[str],
                               right_context: List[str],
                               max_results: int = 100,
                               scoring: str = 'cosine',
                               seed_side = 'both',  # or 'left', 'right'
                               trace: bool = False) -> List[Tuple[str, float]]:
        """
        Find contextually-grounded candidates using GPU acceleration.
        1. Union of reverse lookups (candidates may share SOME context)
        2. Score each candidate with total PMI to context
        3. Filter by total PMI threshold
        4. Score remaining with selected method
        5. Return top-k
        """

        candidates = set()
        if seed_side in ('left', 'both') and left_context:
            for word in left_context:
                if word in self.left_word_to_idx:
                    candidates |= self.left_word_to_idx[word]
        if seed_side in ('right', 'both') and right_context:
            for word in right_context:
                if word in self.right_word_to_idx:
                    candidates |= self.right_word_to_idx[word]

        if not self.gpu_ready:
            return []

        # Unknown units return empty - parser will split them
        if unit_text not in self.unit_to_idx:
            if trace:
                print(f"\n TRACE: '{unit_text}' not in index - will be split")
            return []

        target_idx = self.unit_to_idx[unit_text]
        target_length = unit_text.count(' ') + 1
        if trace:
            print(f"\n TRACE: gpu_contextual_candidates('{unit_text}')")
            print(f" left_context: {left_context}")
            print(f" right_context: {right_context}")
            print(f" Known unit (idx={target_idx})")

        # Get length-filtered indices
        length_indices = set(self.idx_by_length.get(target_length, []))
        if not length_indices:
            if trace:
                print(f" No units of length {target_length}")
            return []

        if trace:
            print(f" Units of length {target_length}: {len(length_indices)}")

        # Step 1: Union of candidates sharing ANY context words
        candidates = set()
        for word in left_context:
            if word in self.left_word_to_idx:
                candidates |= self.left_word_to_idx[word]
        for word in right_context:
            if word in self.right_word_to_idx:
                candidates |= self.right_word_to_idx[word]

        if trace:
            print(f" Union candidates: {len(candidates)}")

        # Step 2: Score each candidate with total PMI to context words

        cond_prob_scores = {}
        for cand_idx in candidates:
            cand_text = self.unit_list[cand_idx]
            pattern = self.units.get(cand_text)
            if not pattern:
                cond_prob_scores[cand_idx] = 0.0
                continue

            score = 0.0
            left_total = sum(pattern.left_words.values()) or 1
            right_total = sum(pattern.right_words.values()) or 1

            for word in left_context:
                if word in pattern.left_words:
                    score += pattern.left_words[word] / left_total  # P(word | cand left)

            for word in right_context:
                if word in pattern.right_words:
                    score += pattern.right_words[word] / right_total  # P(word | cand right)

            cond_prob_scores[cand_idx] = score

        # Step 3 Filter by minimum total cond prob
        #min_cond_prob = 0.2  # tune: sum P >=0.2; lower=more candidates
        #OOM error. Test stricter PMI
        min_cond_prob = 0.5  # tune: sum P >0.5; lower=more candidates
        candidates = {idx for idx, prob in cond_prob_scores.items() if prob >= min_cond_prob}

        if trace:
            print(f" Candidates after cond prob filter (min={min_cond_prob}): {len(candidates)}")

        # Filter by length and remove target
        candidates = candidates & length_indices
        candidates.discard(target_idx)

        if trace:
            print(f" After length filter: {len(candidates)}")

        if not candidates:
            return []

        # Proceed with scoring (original code from here)
        candidate_list = list(candidates)
        if scoring == 'pmi':
            # Use the pre-computed PMI scores as final (since we're PMI-focused)
            combined_scores = torch.tensor([pmi_scores[idx] for idx in candidate_list], dtype=torch.float32, device=DEVICE)
        else:
            # GPU-based cosine or ic_cosine - use batching to avoid memory issues
            batch_size = 1000  # Process candidates in batches
            target_left = self.gpu_left[target_idx].float()
            target_right = self.gpu_right[target_idx].float()

            all_scores = []
            for batch_start in range(0, len(candidate_list), batch_size):
                batch_end = min(batch_start + batch_size, len(candidate_list))
                batch_indices = candidate_list[batch_start:batch_end]
                batch_indices_t = torch.tensor(batch_indices, dtype=torch.long, device=DEVICE)

                cand_left = self.gpu_left[batch_indices_t].float()
                cand_right = self.gpu_right[batch_indices_t].float()

                if scoring == 'ic_cosine':
                    ic_left = self.gpu_ic_left.float()
                    ic_right = self.gpu_ic_right.float()
                    weighted_target_left = target_left * ic_left
                    weighted_target_right = target_right * ic_right
                    left_scores = torch.mv(cand_left, weighted_target_left)
                    right_scores = torch.mv(cand_right, weighted_target_right)
                else:
                    left_scores = torch.mv(cand_left, target_left)
                    right_scores = torch.mv(cand_right, target_right)
                batch_scores = (left_scores + right_scores) / 2.0
                all_scores.extend(batch_scores.cpu().tolist())

            combined_scores = torch.tensor(all_scores, dtype=torch.float32, device=DEVICE)

        if trace:
            print(f" Scored {len(candidate_list)} candidates using {scoring}")

        # Apply consensus weighting: boost candidates whose overlaps are shared by many others
        # This creates "winner takes all" dynamics where common overlaps dominate
        # VECTORIZED: Uses matrix operations instead of Python loops for efficiency

        # Step 1: Build overlap matrix (N_candidates x N_words)
        # Entry [i, j] = 1 if candidate i has context word j in its pattern
        all_context_words = set(left_context) | set(right_context)
        word_to_idx = {word: idx for idx, word in enumerate(sorted(all_context_words))}
        n_candidates = len(candidate_list)
        n_words = len(word_to_idx)

        # Initialize overlap matrix and overlap counts
        overlap_matrix = torch.zeros((n_candidates, n_words), device=DEVICE, dtype=torch.float32)
        overlap_counts = torch.ones(n_candidates, device=DEVICE, dtype=torch.float32)  # avoid div by 0

        for i, cand_idx in enumerate(candidate_list):
            cand_text = self.unit_list[cand_idx]
            pattern = self.units.get(cand_text)
            if not pattern:
                continue

            num_overlaps = 0
            for word in left_context:
                if word in pattern.left_words and word in word_to_idx:
                    j = word_to_idx[word]
                    overlap_matrix[i, j] = 1.0
                    num_overlaps += 1
            for word in right_context:
                if word in pattern.right_words and word in word_to_idx:
                    j = word_to_idx[word]
                    if overlap_matrix[i, j] == 0:  # avoid double-counting
                        overlap_matrix[i, j] = 1.0
                        num_overlaps += 1

            if num_overlaps > 0:
                overlap_counts[i] = num_overlaps

        # Step 2: Count how many candidates share each context word
        # Shape: (N_words,) - counts across all candidates for each word
        word_frequency = overlap_matrix.sum(dim=0)

        # Step 3: Create frequency weight vector using log
        # freq_weights[j] = log(count_of_word_j + 1)
        freq_weights = torch.log(word_frequency + 1.0)

        # Step 4: Vectorized consensus computation
        # consensus_scores = overlap_matrix @ freq_weights / overlap_counts
        # This computes: for each candidate, sum of log-frequencies of its overlaps, divided by count
        if n_words > 0:
            consensus_scores = torch.mv(overlap_matrix, freq_weights) / overlap_counts
        else:
            consensus_scores = torch.zeros(n_candidates, device=DEVICE, dtype=torch.float32)

        if trace:
            print(f" Consensus weights: min={consensus_scores.min().item():.3f}, max={consensus_scores.max().item():.3f}")

        # Apply multiplicative boost: base_score * (1 + consensus_weight)
        # This implements "winner takes all": candidates with high-consensus overlaps dominate
        combined_scores = combined_scores * (1.0 + consensus_scores)

        if trace:
            print(f" After consensus weighting: min={combined_scores.min().item():.3f}, max={combined_scores.max().item():.3f}")

        candidate_indices = torch.tensor(candidate_list, dtype=torch.long, device=DEVICE)
        k = min(max_results, len(candidates))
        top_scores, top_indices = torch.topk(combined_scores, k)

        results = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
            unit_idx = candidate_indices[idx].item()
            results.append((self.unit_list[unit_idx], score))

        if trace:
            print(f" Top 5 results:")
            for text, score in results[:5]:
                print(f" {score:.3f}: '{text}'")

        return results


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
        query_left_f = query_left.float()
        query_right_f = query_right.float()

        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]

            cand_indices = torch.tensor([self.unit_to_idx[t] for t in batch_candidates],
                                        dtype=torch.long, device=DEVICE)

            # Gather candidate vectors and compute similarity
            with torch.no_grad():
                cand_left = self.gpu_left[cand_indices].float()
                cand_right = self.gpu_right[cand_indices].float()

                # Cosine similarity (vectors are normalized)
                left_sim = torch.mv(cand_left, query_left_f)
                right_sim = torch.mv(cand_right, query_right_f)
                avg_sim = (left_sim + right_sim) / 2.0

                # Filter by threshold within this batch
                mask = avg_sim >= threshold
                batch_matching_indices = torch.where(mask)[0]
                batch_matching_sims = avg_sim[mask]

                for i, sim in zip(batch_matching_indices.cpu().tolist(),
                                  batch_matching_sims.cpu().tolist()):
                    all_results.append((batch_candidates[i], sim))

                # Free GPU memory
                del cand_left, cand_right, left_sim, right_sim, avg_sim

            # Clear MPS cache periodically
            if batch_start > 0 and batch_start % (batch_size * 5) == 0:
                torch.mps.empty_cache()

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

    def gpu_batch_expansion_sizes(self, unit_texts: List[str], threshold: float = 0.3,
                                    same_length: bool = True) -> Dict[str, int]:
        """
        FAST: Compute expansion SIZES for multiple units in ONE matrix operation.

        This is much faster than full expansion because:
        1. Single pass - no iterative expansion
        2. Only counts matches - no tracking of actual members
        3. Batched matrix multiply per length group

        Returns:
            Dict mapping each unit to its expansion size (count of similar units)
        """
        if not self.gpu_ready:
            # Fall back to sequential
            result = {}
            for unit in unit_texts:
                similar = self.gpu_find_similar(unit, threshold, same_length, 1000)
                result[unit] = 1 + len(similar)  # Include self
            return result

        # Clear GPU cache
        torch.mps.empty_cache()

        # Initialize all units with size 1 (themselves)
        result = {u: 1 for u in unit_texts}

        # Filter to units in our index
        valid_units = [u for u in unit_texts if u in self.unit_to_idx]
        if not valid_units:
            return result

        # Group queries by length
        queries_by_length = {}
        for u in valid_units:
            length = u.count(' ') + 1
            if length not in queries_by_length:
                queries_by_length[length] = []
            queries_by_length[length].append(u)

        # Process each length group
        for length, length_queries in queries_by_length.items():
            if length not in self.units_by_length:
                continue

            # Get candidates of this length
            length_cands = [c for c in self.units_by_length[length] if c in self.unit_to_idx]
            if not length_cands:
                continue

            # Build query tensor
            q_indices = torch.tensor([self.unit_to_idx[q] for q in length_queries],
                                    dtype=torch.long, device=DEVICE)

            # Process candidates in large batches (can be bigger since we only need counts)
            cand_batch_size = 10000
            match_counts = np.zeros(len(length_queries), dtype=np.int32)

            for cand_start in range(0, len(length_cands), cand_batch_size):
                cand_end = min(cand_start + cand_batch_size, len(length_cands))
                batch_cands = length_cands[cand_start:cand_end]

                c_indices = torch.tensor([self.unit_to_idx[c] for c in batch_cands],
                                        dtype=torch.long, device=DEVICE)

                with torch.no_grad():
                    q_left = self.gpu_left[q_indices].float()
                    q_right = self.gpu_right[q_indices].float()
                    c_left = self.gpu_left[c_indices].float()
                    c_right = self.gpu_right[c_indices].float()

                    # Matrix multiply: (n_q, vocab) @ (vocab, n_c) -> (n_q, n_c)
                    left_sim = torch.mm(q_left, c_left.T)
                    right_sim = torch.mm(q_right, c_right.T)
                    avg_sim = (left_sim + right_sim) / 2.0

                    # Count matches per query (sum of booleans)
                    batch_counts = (avg_sim >= threshold).sum(dim=1).cpu().numpy()
                    match_counts += batch_counts

                    del q_left, q_right, c_left, c_right, left_sim, right_sim, avg_sim

                torch.mps.empty_cache()

            # Update results
            for qi, query in enumerate(length_queries):
                result[query] = 1 + int(match_counts[qi])  # +1 for self

        return result

    def gpu_batch_expand(self, unit_texts: List[str], threshold: float = 0.3,
                         same_length: bool = True, max_per_unit: int = 100,
                         max_iters: int = 2) -> Dict[str, Set[str]]:
        """
        Compute bidirectional expansions for MULTIPLE units in ONE matrix operation.

        This is the key optimization: instead of processing spans one by one,
        we process ALL spans simultaneously using matrix multiplication.

        Args:
            unit_texts: List of units to expand
            threshold: Minimum similarity for expansion
            same_length: Only expand to same-length units
            max_per_unit: Max expansions per unit
            max_iters: Number of expansion iterations

        Returns:
            Dict mapping each input unit to its expansion set
        """
        if not self.gpu_ready:
            # Fall back to sequential processing
            result = {}
            for unit in unit_texts:
                similar = self.gpu_find_similar(unit, threshold, same_length, max_per_unit)
                result[unit] = {unit} | {s[0] for s in similar}
            return result

        # Clear GPU cache before batch operation
        torch.mps.empty_cache()

        # Filter to units that exist in our GPU index
        valid_units = [u for u in unit_texts if u in self.unit_to_idx]
        if not valid_units:
            return {u: {u} for u in unit_texts}

        # Get query indices
        query_indices = torch.tensor([self.unit_to_idx[u] for u in valid_units],
                                     dtype=torch.long, device=DEVICE)

        # Get query vectors - shape: (n_queries, vocab_size)
        query_left = self.gpu_left[query_indices]
        query_right = self.gpu_right[query_indices]

        # Initialize expansions with the units themselves
        expansions = {u: {u} for u in unit_texts}

        # Track which units we've already found (to avoid re-adding)
        all_found = {u: {u} for u in valid_units}

        for iteration in range(max_iters):
            # Collect all current expansion members that need similarity search
            current_queries = []
            query_to_original = []  # Map back to original unit

            for orig_unit in valid_units:
                for member in all_found[orig_unit]:
                    if member in self.unit_to_idx:
                        current_queries.append(member)
                        query_to_original.append(orig_unit)

            if not current_queries:
                break

            # Get unique queries (avoid duplicate computation)
            unique_queries = list(set(current_queries))
            unique_indices = torch.tensor([self.unit_to_idx[u] for u in unique_queries],
                                          dtype=torch.long, device=DEVICE)

            # Get query vectors for this iteration
            iter_query_left = self.gpu_left[unique_indices]  # (n_unique, vocab)
            iter_query_right = self.gpu_right[unique_indices]

            # Group candidates by length if needed
            if same_length:
                # Process each length group separately
                new_found = {u: set() for u in valid_units}

                for length, length_units in self.units_by_length.items():
                    # Get queries of this length
                    length_queries = [q for q in unique_queries
                                     if q.count(' ') + 1 == length]
                    if not length_queries:
                        continue

                    # Get candidates of this length
                    length_cands = [c for c in length_units if c in self.unit_to_idx]
                    if not length_cands:
                        continue

                    # Build query tensors (queries are usually small)
                    q_indices = torch.tensor([self.unit_to_idx[q] for q in length_queries],
                                            dtype=torch.long, device=DEVICE)

                    # Build query-to-original mapping for this length
                    query_to_orig_map = {}
                    for qi, query in enumerate(length_queries):
                        query_to_orig_map[qi] = []
                        for cq, orig in zip(current_queries, query_to_original):
                            if cq == query:
                                query_to_orig_map[qi].append(orig)

                    # Process candidates in batches to avoid OOM
                    cand_batch_size = 2000
                    for cand_start in range(0, len(length_cands), cand_batch_size):
                        cand_end = min(cand_start + cand_batch_size, len(length_cands))
                        batch_cands = length_cands[cand_start:cand_end]

                        c_indices = torch.tensor([self.unit_to_idx[c] for c in batch_cands],
                                                dtype=torch.long, device=DEVICE)

                        with torch.no_grad():
                            q_left = self.gpu_left[q_indices].float()   # (n_q, vocab)
                            q_right = self.gpu_right[q_indices].float()
                            c_left = self.gpu_left[c_indices].float()   # (n_c, vocab)
                            c_right = self.gpu_right[c_indices].float()

                            # MATRIX MULTIPLY: compute ALL similarities at once
                            # (n_q, vocab) @ (vocab, n_c) -> (n_q, n_c)
                            left_sim = torch.mm(q_left, c_left.T)
                            right_sim = torch.mm(q_right, c_right.T)
                            avg_sim = (left_sim + right_sim) / 2.0

                            # Find matches above threshold
                            matches = (avg_sim >= threshold).cpu().numpy()

                            # Clean up GPU memory
                            del q_left, q_right, c_left, c_right, left_sim, right_sim, avg_sim

                        torch.mps.empty_cache()

                        # Map matches back to original units
                        for qi in range(len(length_queries)):
                            for ci in range(len(batch_cands)):
                                if matches[qi, ci]:
                                    cand = batch_cands[ci]
                                    for orig in query_to_orig_map[qi]:
                                        if cand not in all_found[orig]:
                                            new_found[orig].add(cand)

                # Update expansions with newly found units
                for orig in valid_units:
                    all_found[orig].update(new_found[orig])
                    expansions[orig].update(new_found[orig])
            else:
                # No length filtering - compute against all candidates in batches
                # Build query-to-original mapping
                query_to_orig_map = {}
                for qi, query in enumerate(unique_queries):
                    query_to_orig_map[qi] = []
                    for cq, orig in zip(current_queries, query_to_original):
                        if cq == query:
                            query_to_orig_map[qi].append(orig)

                cand_batch_size = 2000
                for cand_start in range(0, len(self.unit_list), cand_batch_size):
                    cand_end = min(cand_start + cand_batch_size, len(self.unit_list))
                    batch_cands = self.unit_list[cand_start:cand_end]

                    c_indices = torch.tensor(list(range(cand_start, cand_end)),
                                            dtype=torch.long, device=DEVICE)

                    with torch.no_grad():
                        q_left = iter_query_left.float()
                        q_right = iter_query_right.float()
                        c_left = self.gpu_left[c_indices].float()
                        c_right = self.gpu_right[c_indices].float()

                        left_sim = torch.mm(q_left, c_left.T)
                        right_sim = torch.mm(q_right, c_right.T)
                        avg_sim = (left_sim + right_sim) / 2.0

                        matches = (avg_sim >= threshold).cpu().numpy()

                        del q_left, q_right, c_left, c_right, left_sim, right_sim, avg_sim

                    torch.mps.empty_cache()

                    for qi in range(len(unique_queries)):
                        for ci in range(len(batch_cands)):
                            if matches[qi, ci]:
                                cand = batch_cands[ci]
                                for orig in query_to_orig_map[qi]:
                                    if cand not in all_found[orig]:
                                        all_found[orig].add(cand)
                                        expansions[orig].add(cand)

        # Limit expansions per unit
        for unit in expansions:
            if len(expansions[unit]) > max_per_unit:
                # Keep original unit + top matches (we don't have scores here, so just truncate)
                exp_list = list(expansions[unit])
                expansions[unit] = {unit} | set(exp_list[:max_per_unit])

        return expansions

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


def bidirectional_expansion_contextual(unit_text: str,
                                       current_left: Counter,
                                       current_right: Counter,
                                       catalog: UnitCatalog,
                                       max_results: int = 100,
                                       max_iters: int = 2) -> Tuple[Set[str], Set[str]]:
    """
    GPU-accelerated contextual bidirectional expansion.

    Key insight: Start from presented context (intersection filtering),
    then score by similarity to target using GPU matrix operations.

    1. Intersection of reverse lookups → candidates sharing presented context
    2. GPU matrix-vector multiply → overlap scores with target
    3. Top-k selection → niche grouping for this sentence

    Returns: (unit_expansion, context_pattern_expansion)
    """
    target_unit = catalog.get_unit(unit_text)
    if not target_unit:
        return (set(), set())

    # Initialize
    unit_expansion = {unit_text}

    # Build context pattern for target
    target_pattern = frozenset(target_unit.left_words.most_common(5) +
                               target_unit.right_words.most_common(5))
    context_expansion = {target_pattern}

    # Convert counters to lists for GPU method
    left_ctx_list = list(current_left.keys())
    right_ctx_list = list(current_right.keys())

    for iteration in range(max_iters):
        # ============================================================
        # GPU-ACCELERATED CONTEXTUAL EXPANSION
        # ============================================================
        # Find units that:
        # 1. Share the presented context (intersection of reverse lookups)
        # 2. Are most similar to target (GPU scoring)

        new_candidates = []

        # Expand from each unit in current expansion
        for existing_unit in list(unit_expansion):
            candidates = catalog.gpu_contextual_candidates(
                existing_unit,
                left_context=left_ctx_list,
                right_context=right_ctx_list,
                max_results=max_results
            )
            new_candidates.extend(candidates)

        # Deduplicate and filter already-expanded
        seen = {}
        for text, score in new_candidates:
            if text not in unit_expansion:
                if text not in seen or score > seen[text]:
                    seen[text] = score

        # Add top candidates to expansion
        sorted_candidates = sorted(seen.items(), key=lambda x: -x[1])[:max_results]
        new_units = {text for text, _ in sorted_candidates}

        if not new_units:
            break

        unit_expansion.update(new_units)

        # Build context patterns for new units
        for u in new_units:
            u_unit = catalog.get_unit(u)
            if u_unit:
                pattern = frozenset(u_unit.left_words.most_common(5) +
                                   u_unit.right_words.most_common(5))
                context_expansion.add(pattern)

    return (unit_expansion, context_expansion)

def compute_bidir_energy(span: Span,
                         left_context,
                         right_context,
                         catalog: UnitCatalog,
                         debug=False,
                         use_contextual: bool = True) -> Tuple[float, int, int, Set[str], Set]:
    """Energy via bidirectional expansion.
    Args:
        use_contextual: If True, use contextual expansion (local context-driven).
                       If False, use simple expansion (global statistics).
    Returns:
        Tuple of (energy, unit_exp_size, ctx_exp_size, unit_exp_set, ctx_exp_set)
    """
    current_left = Counter(left_context)
    current_right = Counter(right_context)
    if use_contextual:
        unit_exp, ctx_exp = bidirectional_expansion_contextual(
            span.text, current_left, current_right, catalog
        )
    else:
        unit_exp, ctx_exp = bidirectional_expansion_simple(
            span.text, current_left, current_right, catalog
        )
    unit_exp_size = len(unit_exp)
    ctx_exp_size = len(ctx_exp)
    unit_exp_set = unit_exp  # The set of substitutes

    if unit_exp_size == 0:
        return (100.0, 0, 0, set(), set())

    # === ORIGINAL combined (generalizability) ===
    combined = unit_exp_size * math.log(ctx_exp_size + 1)

    # === NEW: Prediction consensus term (low entropy = high agreement = strong prediction) ===
    prediction_consensus = 0.0
    if unit_exp_set:
        # Aggregate right_words across all substitutes
        agg_right = Counter()
        for sub in unit_exp_set:
            pat = catalog.get_unit(sub)
            if pat and pat.right_words:
                agg_right.update(pat.right_words)

        if agg_right:
            total = sum(agg_right.values())
            if total > 0:
                # Shannon entropy of the normalized distribution
                entropy = -sum(
                    (count / total) * math.log2((count / total) + 1e-10)
                    for count in agg_right.values()
                )
                # Normalize (max entropy is log2(num distinct words))
                max_entropy = math.log2(len(agg_right) + 1e-10) if len(agg_right) > 1 else 0
                norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
                prediction_consensus = 1.0 - norm_entropy  # 1.0 = perfect agreement

    # === Combine: weight consensus by evidence size (U) ===
    pred_term = 2.0 * prediction_consensus * math.log(unit_exp_size + 1)  # tune 2.0

    # === Final energy ===
    energy = -math.log(combined + pred_term + 1)

    if debug and span.length >= 2:
        print(f"  \"{span.text}\": U={unit_exp_size}, C={ctx_exp_size}, "
              f"Consensus={prediction_consensus:.3f}, Pred_term={pred_term:.3f}, E={energy:.3f}")
        if unit_exp_size <= 10:
            print(f"    expansion: {list(unit_exp)[:10]}")

    return (energy, unit_exp_size, ctx_exp_size, unit_exp_set, ctx_exp)

class IncrementalBidirParser:
    """
    Incremental bottom-up chart parser.

    Builds parse incrementally as words arrive, computing each span exactly once.
    Uses O(n²) time instead of exponential, with natural caching via chart storage.
    Supports both batch parsing and true streaming for neurosimulator applications.
    """

    def __init__(self, catalog, context_window=3, debug=False):
        self.catalog = catalog
        self.context_window = context_window
        self.debug = debug
        self.chart = {}  # (start, end) -> ParseNode
        self.tokens = []

    def reset(self):
        """Clear the chart and tokens for a new parse."""
        self.chart = {}
        self.tokens = []

    def add_word(self, word: str):
        """
        Add a word to the parse incrementally.
        Creates all new spans ending at this position and updates the chart.
        """
        self.tokens.append(word.lower())
        n = len(self.tokens)
        k = n - 1  # Index of new word

        # Create all spans ending at position n (after the new word)
        for i in range(n):
            self._compute_span(i, n)

    def _compute_span(self, i: int, j: int):
        """
        Compute best parse for span [i, j) using DP.
        Assumes all smaller spans are already in the chart.
        """
        if (i, j) in self.chart:
            return  # Already computed

        span = Span(i, j, self.tokens[i:j])

        # Get context for this span
        left_ctx = self.tokens[max(0, i - self.context_window):i]
        right_ctx = self.tokens[j:min(len(self.tokens), j + self.context_window)]

        # Option 1: Treat as a unit
        unit_energy, u_exp, c_exp, u_exp_set, c_exp_set = compute_bidir_energy(
            span, left_ctx, right_ctx, self.catalog, self.debug,
            use_contextual=True
        )

        # Option 2: Try all binary splits (if length >= 2)
        best_split_energy = float('inf')
        best_split = None

        if j - i >= 2:
            for m in range(i + 1, j):
                # Left and right children should already be in chart
                left_child = self.chart.get((i, m))
                right_child = self.chart.get((m, j))

                if left_child and right_child:
                    split_energy = left_child.energy + right_child.energy + 2.0

                    if split_energy < best_split_energy:
                        best_split_energy = split_energy
                        best_split = (m, left_child, right_child)

        # Choose best option
        if best_split is None or unit_energy <= best_split_energy:
            node = ParseNode(span, unit_energy)
            node.unit_expansion = u_exp
            node.context_expansion = c_exp
            node.unit_expansion_set = u_exp_set
            node.context_expansion_set = c_exp_set
            self.chart[(i, j)] = node
        else:
            m, left_child, right_child = best_split
            node = ParseNode(span, best_split_energy, split_point=m,
                           left=left_child, right=right_child)
            self.chart[(i, j)] = node

    def get_current_parse(self) -> ParseNode:
        """Get the best parse for all tokens seen so far."""
        if not self.tokens:
            return None
        return self.chart.get((0, len(self.tokens)))

    def parse(self, tokens: List[str]) -> ParseNode:
        """
        Parse a complete sequence of tokens (batch mode).
        Equivalent to adding all words then getting the final parse.
        """
        self.reset()
        for token in tokens:
            self.add_word(token)
        return self.get_current_parse()


class SimpleBidirParser:
    """Parser using simplified bidirectional expansion."""

    def __init__(self, catalog, context_window=3, debug=False, use_contextual=True):
        self.catalog = catalog
        self.context_window = context_window
        self.debug = debug
        self.use_contextual = use_contextual

    def parse(self, tokens):
        span = Span(0, len(tokens), tokens)
        return self._parse_span(span, tokens)

    def _parse_span(self, span, full_tokens):
        left_ctx = full_tokens[max(0, span.start - self.context_window):span.start]
        right_ctx = full_tokens[span.end:min(len(full_tokens), span.end + self.context_window)]

        # Test as unit
        unit_energy, u_exp, c_exp = compute_bidir_energy(
            span, left_ctx, right_ctx, self.catalog, self.debug,
            use_contextual=self.use_contextual
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

    def parse_batch(self, tokens):
        """
        Parse using batch GPU computation for ALL spans simultaneously.

        Instead of computing expansions one span at a time (slow),
        this method:
        1. Enumerates ALL possible spans upfront
        2. Computes ALL expansions in ONE batch GPU operation
        3. Computes energies from pre-computed expansions
        4. Does CKY-style DP using pre-computed energies

        This should be MUCH faster than the recursive approach.
        """
        n = len(tokens)
        if n == 0:
            return None

        # Step 1: Enumerate all possible spans
        all_spans = []
        span_to_idx = {}
        for i in range(n):
            for j in range(i + 1, n + 1):
                span_text = " ".join(tokens[i:j])
                span = Span(i, j, tokens[i:j])
                span_to_idx[(i, j)] = len(all_spans)
                all_spans.append((span, span_text))

        # Step 2: Batch compute all expansion SIZES using GPU (fast single-pass)
        span_texts = [text for _, text in all_spans]
        expansion_sizes = self.catalog.gpu_batch_expansion_sizes(
            span_texts,
            threshold=0.30,
            same_length=True
        )

        # Step 3: Compute energy for each span from its expansion size
        span_energies = {}
        span_exp_sizes = {}
        for (span, span_text) in all_spans:
            unit_exp_size = expansion_sizes.get(span_text, 1)

            # Context expansion size approximated by unit expansion size
            # (since similar units have similar contexts)
            ctx_exp_size = unit_exp_size

            if unit_exp_size == 0:
                energy = 100.0
            else:
                combined = unit_exp_size * math.log(ctx_exp_size + 1)
                energy = -math.log(combined + 1)

            span_energies[(span.start, span.end)] = energy
            span_exp_sizes[(span.start, span.end)] = (unit_exp_size, ctx_exp_size)

        # Step 4: CKY-style DP to find best parse
        # best[i][j] = (energy, ParseNode) for span [i, j)
        best = {}

        # Base case: single tokens
        for i in range(n):
            j = i + 1
            span = Span(i, j, tokens[i:j])
            energy = span_energies[(i, j)]
            u_exp, c_exp = span_exp_sizes[(i, j)]
            node = ParseNode(span, energy)
            node.unit_expansion = u_exp
            node.context_expansion = c_exp
            best[(i, j)] = (energy, node)

        # Fill in longer spans
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                span = Span(i, j, tokens[i:j])

                # Option 1: treat as single unit
                unit_energy = span_energies[(i, j)]
                u_exp, c_exp = span_exp_sizes[(i, j)]

                # Option 2: best split
                best_split_energy = float('inf')
                best_split = None
                for m in range(i + 1, j):
                    left_energy, left_node = best[(i, m)]
                    right_energy, right_node = best[(m, j)]
                    split_energy = left_energy + right_energy + 2.0
                    if split_energy < best_split_energy:
                        best_split_energy = split_energy
                        best_split = (m, left_node, right_node)

                # Choose best option
                if best_split is None or unit_energy <= best_split_energy:
                    node = ParseNode(span, unit_energy)
                    node.unit_expansion = u_exp
                    node.context_expansion = c_exp
                    best[(i, j)] = (unit_energy, node)
                else:
                    m, left_node, right_node = best_split
                    node = ParseNode(span, best_split_energy, split_point=m,
                                   left=left_node, right=right_node)
                    best[(i, j)] = (best_split_energy, node)

        # Return the parse for the full span
        _, root = best[(0, n)]
        return root

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
