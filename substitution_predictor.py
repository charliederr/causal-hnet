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
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# PyTorch for GPU acceleration (MPS on Apple Silicon)
try:
    import torch
    HAS_TORCH = True
    # Check for MPS (Metal Performance Shaders) on Apple Silicon
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print(f"GPU acceleration: Using Apple Metal (MPS)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"GPU acceleration: Using CUDA")
    else:
        DEVICE = torch.device("cpu")
        print(f"GPU acceleration: Not available, using CPU")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("PyTorch not installed - using CPU-only mode")

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
class UnitPrediction:
    """A predicted multi-token unit with score and explanation."""
    unit_text: str
    tokens: List[str]
    score: float
    context_overlap: float  # How well unit's left-context matches prompt
    frequency: int  # How often this unit appears in corpus


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
                 min_unit_freq: int = 10,  # Raised from 5
                 share_gpu_with_catalog: bool = True):
        self.catalog = catalog
        self.context_window = context_window
        self.top_k_context = top_k_context
        self.min_unit_freq = min_unit_freq
        self.share_gpu = share_gpu_with_catalog

        # Build reverse index: left_word -> units that have this left context
        self._build_indices()

    def _build_indices(self):
        """Build indices for fast lookup."""
        print("Building prediction indices...")

        # Index: context word -> units with that context
        self.left_context_index = defaultdict(set)
        self.right_context_index = defaultdict(set)  # For finding substitution classes

        # Track word document frequency for IDF weighting
        self.word_doc_freq = Counter()  # How many units have this word in context

        # Store right-context distributions for prediction
        self.unit_right_contexts = {}

        # Store left-context distributions (normalized) for better matching
        self.unit_left_contexts = {}

        # Store raw context patterns for substitution class computation
        self.unit_patterns = {}

        # Filter to frequent units only for speed
        freq_units = {
            text: pattern for text, pattern in self.catalog.units.items()
            if pattern.count >= self.min_unit_freq
        }
        print(f"  Filtering to {len(freq_units):,} units with freq >= {self.min_unit_freq}")

        total_units = len(freq_units)

        for i, (unit_text, pattern) in enumerate(freq_units.items()):
            if i % 5000 == 0:
                print(f"  Processing unit {i:,}...")

            # Store raw pattern for substitution class computation
            self.unit_patterns[unit_text] = pattern

            # Index by top left context words
            top_left = pattern.left_words.most_common(self.top_k_context)
            for word, count in top_left:
                self.left_context_index[word].add(unit_text)
                self.word_doc_freq[word] += 1

            # Index by top right context words (for substitution class lookup)
            top_right = pattern.right_words.most_common(self.top_k_context)
            for word, count in top_right:
                self.right_context_index[word].add(unit_text)

            # Store normalized left-context distribution
            left_total = sum(pattern.left_words.values())
            if left_total > 0:
                self.unit_left_contexts[unit_text] = {
                    word: count / left_total
                    for word, count in top_left
                }

            # Store normalized right-context distribution
            right_total = sum(pattern.right_words.values())
            if right_total > 0:
                self.unit_right_contexts[unit_text] = {
                    word: count / right_total
                    for word, count in top_right
                }

        # Compute IDF weights
        self.word_idf = {}
        for word, doc_freq in self.word_doc_freq.items():
            # IDF = log(N / df), capped to avoid extreme values
            self.word_idf[word] = min(math.log(total_units / (doc_freq + 1)) + 1, 5.0)

        print(f"  Indexed {len(self.unit_right_contexts):,} units")
        print(f"  Left context vocabulary: {len(self.left_context_index):,} words")

        # Build vectorized representations for fast batch similarity computation
        print("  Building vectorized context representations...")
        self._build_vectorized_contexts()
        print(f"  Vectorized {len(self.unit_list)} units with {len(self.vocab_to_idx)} vocabulary")

    def _build_vectorized_contexts(self):
        """
        Build GPU tensors for fast batch similarity computation.

        Instead of computing Jaccard similarity one pair at a time,
        we compute ALL similarities in parallel using GPU matrix operations.
        This transforms a 60-second operation into milliseconds.
        """
        # Build vocabulary index
        all_context_words = set()
        for pattern in self.unit_patterns.values():
            all_context_words.update(w for w, _ in pattern.left_words.most_common(20))
            all_context_words.update(w for w, _ in pattern.right_words.most_common(20))

        self.vocab_to_idx = {word: i for i, word in enumerate(sorted(all_context_words))}
        vocab_size = len(self.vocab_to_idx)

        # Build unit list and matrices
        self.unit_list = list(self.unit_patterns.keys())
        self.unit_to_idx = {unit: i for i, unit in enumerate(self.unit_list)}
        n_units = len(self.unit_list)

        # Sparse representation for CPU fallback
        self.left_context_topk = {}  # unit_idx -> (word_indices, values)
        self.right_context_topk = {}

        for unit_text, pattern in self.unit_patterns.items():
            unit_idx = self.unit_to_idx[unit_text]

            # Left context top-k
            left_items = pattern.left_words.most_common(10)
            if left_items:
                indices = np.array([self.vocab_to_idx.get(w, 0) for w, _ in left_items])
                values = np.array([c for _, c in left_items], dtype=np.float32)
                values = values / (values.sum() + 1e-10)  # Normalize
                self.left_context_topk[unit_idx] = (indices, values)

            # Right context top-k
            right_items = pattern.right_words.most_common(10)
            if right_items:
                indices = np.array([self.vocab_to_idx.get(w, 0) for w, _ in right_items])
                values = np.array([c for _, c in right_items], dtype=np.float32)
                values = values / (values.sum() + 1e-10)
                self.right_context_topk[unit_idx] = (indices, values)

        # Build GPU tensors if PyTorch is available
        if HAS_TORCH and DEVICE is not None:
            # Check if we can share GPU tensors with catalog
            if self.share_gpu and hasattr(self.catalog, 'gpu_ready') and self.catalog.gpu_ready:
                print(f"  Sharing GPU tensors with catalog...")
                self._share_catalog_gpu_tensors()
            else:
                print(f"  Building GPU tensors on {DEVICE}...")
                self._build_gpu_tensors(vocab_size, n_units)

    def _share_catalog_gpu_tensors(self):
        """
        Share GPU tensors with the catalog instead of building our own.
        This saves ~12GB of GPU memory.
        """
        # Use catalog's GPU tensors directly
        self.gpu_left = self.catalog.gpu_left
        self.gpu_right = self.catalog.gpu_right
        self.n_units = len(self.catalog.unit_list)
        self.compact_vocab_size = self.gpu_left.shape[1]

        # Build unit index mapping from our unit list to catalog's
        # (they should be the same, but let's be safe)
        self.unit_to_gpu_idx = {}
        for unit_text in self.unit_patterns:
            if unit_text in self.catalog.unit_to_idx:
                self.unit_to_gpu_idx[unit_text] = self.catalog.unit_to_idx[unit_text]

        self.gpu_available = True
        mem_mb = (self.gpu_left.numel() + self.gpu_right.numel()) * 2 / (1024 * 1024)
        print(f"  Shared GPU matrices: {self.n_units} units × {self.compact_vocab_size} vocab ({mem_mb:.1f} MB)")

    def _build_gpu_tensors(self, vocab_size: int, n_units: int):
        """
        Build dense GPU tensors and keep them on GPU permanently.

        Key insight: Avoid CPU-GPU transfers during computation.
        Keep everything on GPU, use indexed operations.

        Memory: 70k × 47k × 2 bytes (float16) = ~6.5GB per matrix
        Total: ~13GB - fits in M4's unified memory
        """
        # Use reduced vocabulary: only words that appear in contexts
        effective_vocab = set()
        for unit_idx in range(n_units):
            if unit_idx in self.left_context_topk:
                indices, _ = self.left_context_topk[unit_idx]
                effective_vocab.update(indices.tolist())
            if unit_idx in self.right_context_topk:
                indices, _ = self.right_context_topk[unit_idx]
                effective_vocab.update(indices.tolist())

        self.compact_vocab = sorted(effective_vocab)
        self.idx_to_compact = {idx: i for i, idx in enumerate(self.compact_vocab)}
        compact_size = len(self.compact_vocab)

        print(f"  Compacting vocabulary: {vocab_size} -> {compact_size} effective words")

        # Build dense matrices on CPU first (float16 to save memory)
        cpu_left = np.zeros((n_units, compact_size), dtype=np.float16)
        cpu_right = np.zeros((n_units, compact_size), dtype=np.float16)

        for unit_idx in range(n_units):
            if unit_idx in self.left_context_topk:
                indices, values = self.left_context_topk[unit_idx]
                for idx, val in zip(indices, values):
                    if idx in self.idx_to_compact:
                        cpu_left[unit_idx, self.idx_to_compact[idx]] = val

            if unit_idx in self.right_context_topk:
                indices, values = self.right_context_topk[unit_idx]
                for idx, val in zip(indices, values):
                    if idx in self.idx_to_compact:
                        cpu_right[unit_idx, self.idx_to_compact[idx]] = val

        # Normalize on CPU
        left_norms = np.linalg.norm(cpu_left.astype(np.float32), axis=1, keepdims=True) + 1e-10
        right_norms = np.linalg.norm(cpu_right.astype(np.float32), axis=1, keepdims=True) + 1e-10
        cpu_left = (cpu_left / left_norms).astype(np.float16)
        cpu_right = (cpu_right / right_norms).astype(np.float16)

        # Transfer to GPU ONCE and keep there permanently
        print(f"  Transferring to GPU ({DEVICE})...")
        self.gpu_left = torch.from_numpy(cpu_left).to(DEVICE)
        self.gpu_right = torch.from_numpy(cpu_right).to(DEVICE)

        # Free CPU memory
        del cpu_left, cpu_right

        self.gpu_available = True
        self.n_units = n_units
        self.compact_vocab_size = compact_size

        mem_mb = (self.gpu_left.numel() + self.gpu_right.numel()) * 2 / (1024 * 1024)
        print(f"  GPU matrices ready: {n_units} units × {compact_size} vocab ({mem_mb:.1f} MB on GPU)")

    def _gpu_batch_similarity(self, query_unit_idx: int,
                               candidate_indices: List[int],
                               batch_size: int = 10000) -> np.ndarray:
        """
        Compute context similarities using GPU matrix operations.

        ALL data stays on GPU - only the final result transfers to CPU.
        Uses indexed gather to select candidate rows from GPU matrices.
        """
        if not hasattr(self, 'gpu_available') or not self.gpu_available:
            return self._cpu_batch_similarity(query_unit_idx, candidate_indices)

        # Create index tensor on GPU
        cand_indices_gpu = torch.tensor(candidate_indices, dtype=torch.long, device=DEVICE)

        # Get query vectors (already on GPU)
        query_left = self.gpu_left[query_unit_idx]  # Shape: (vocab_size,)
        query_right = self.gpu_right[query_unit_idx]

        # Gather candidate vectors (stays on GPU)
        cand_left = self.gpu_left[cand_indices_gpu]  # Shape: (n_candidates, vocab_size)
        cand_right = self.gpu_right[cand_indices_gpu]

        # Compute cosine similarities with matrix-vector multiply (all on GPU)
        # Since vectors are normalized: dot product = cosine similarity
        left_sim = torch.mv(cand_left.float(), query_left.float())  # Convert to float32 for mv
        right_sim = torch.mv(cand_right.float(), query_right.float())

        # Average of left and right similarities
        avg_sim = (left_sim + right_sim) / 2.0

        # Only transfer final result to CPU
        return avg_sim.cpu().numpy()

    def _cpu_batch_similarity(self, query_unit_idx: int,
                               candidate_indices: List[int]) -> np.ndarray:
        """CPU fallback for batch similarity computation."""
        if query_unit_idx not in self.left_context_topk:
            return np.zeros(len(candidate_indices), dtype=np.float32)

        query_left = self.left_context_topk[query_unit_idx][0]
        query_right = self.right_context_topk.get(query_unit_idx, (np.array([]), np.array([])))[0]

        left_indices = [self.left_context_topk.get(i, (np.array([]), None))[0]
                        for i in candidate_indices]
        right_indices = [self.right_context_topk.get(i, (np.array([]), None))[0]
                         for i in candidate_indices]

        left_sims = self._batch_jaccard_similarity(query_left, left_indices)
        right_sims = self._batch_jaccard_similarity(query_right, right_indices)

        return (left_sims + right_sims) / 2.0

    def _batch_jaccard_similarity(self, query_indices: np.ndarray,
                                   candidate_indices_list: List[np.ndarray]) -> np.ndarray:
        """
        Compute Jaccard similarity between a query and multiple candidates.

        Uses set operations on indices for efficiency. CPU fallback.
        """
        query_set = set(query_indices)
        similarities = np.zeros(len(candidate_indices_list), dtype=np.float32)

        for i, cand_indices in enumerate(candidate_indices_list):
            cand_set = set(cand_indices)
            intersection = len(query_set & cand_set)
            union = len(query_set | cand_set)
            similarities[i] = intersection / union if union > 0 else 0.0

        return similarities

    def compute_context_signature(self, tokens: List[str],
                                   position: int) -> ContextSignature:
        """
        Compute the dynamic 'embedding' at a position.
        This aggregates context from the tokens preceding this position.

        Position weighting: Immediate context words get higher weight.
        """
        # Get left context with position weighting
        start = max(0, position - self.context_window)
        left_tokens = tokens[start:position]
        left_words = Counter()

        # Weight by position: closer = higher weight
        for i, token in enumerate(left_tokens):
            # Distance from prediction position (1 = immediate, higher = further)
            distance = len(left_tokens) - i
            # Exponential decay: immediate context weighted ~3x more than distant
            position_weight = math.exp(-0.3 * (distance - 1))
            # Also apply IDF weight if available
            idf_weight = self.word_idf.get(token, 1.0)
            left_words[token] += position_weight * idf_weight

        # Get right context (what we have so far, for partial matching)
        end = min(len(tokens), position + self.context_window)
        right_tokens = tokens[position:end]
        right_words = Counter()

        for i, token in enumerate(right_tokens):
            distance = i + 1
            position_weight = math.exp(-0.3 * (distance - 1))
            idf_weight = self.word_idf.get(token, 1.0)
            right_words[token] += position_weight * idf_weight

        return ContextSignature(
            left_words=left_words,
            right_words=right_words,
            hierarchy_level=0,
            source_units=[]
        )

    def find_matching_units(self, context_sig: ContextSignature,
                           threshold: float = 0.15,
                           max_units: int = 50) -> List[Tuple[str, float]]:
        """
        Find units whose left-context overlaps with current context.
        Uses IDF-weighted cosine similarity for tighter matching.
        Returns: [(unit_text, overlap_score), ...]
        """
        # Find candidate units via index
        candidates = set()
        for word in context_sig.left_words:
            candidates.update(self.left_context_index.get(word, set()))

        # Score candidates by IDF-weighted cosine similarity
        scored = []
        query_words = context_sig.left_words

        # Compute query norm for cosine similarity
        query_norm_sq = sum(v * v for v in query_words.values())
        if query_norm_sq == 0:
            return []
        query_norm = math.sqrt(query_norm_sq)

        for unit_text in candidates:
            unit_left = self.unit_left_contexts.get(unit_text, {})
            if not unit_left:
                continue

            # IDF-weighted cosine similarity
            dot_product = 0.0
            for word, query_weight in query_words.items():
                if word in unit_left:
                    # Unit weight is normalized probability * IDF
                    unit_weight = unit_left[word] * self.word_idf.get(word, 1.0)
                    dot_product += query_weight * unit_weight

            # Unit norm
            unit_norm_sq = sum(
                (prob * self.word_idf.get(w, 1.0)) ** 2
                for w, prob in unit_left.items()
            )
            if unit_norm_sq == 0:
                continue
            unit_norm = math.sqrt(unit_norm_sq)

            similarity = dot_product / (query_norm * unit_norm)

            if similarity >= threshold:
                scored.append((unit_text, similarity))

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

        # 2. Find matching units (higher threshold for tighter matching)
        matching_units = self.find_matching_units(context_sig, threshold=0.15)

        if not matching_units:
            return [Prediction("<unk>", 1.0, [])]

        # 3. Aggregate predictions from matching units
        # Apply IDF weighting to predictions too - downweight common words
        prediction_scores = defaultdict(float)
        prediction_sources = defaultdict(list)

        for unit_text, overlap_score in matching_units:
            right_dist = self.unit_right_contexts.get(unit_text, {})

            # Weight by overlap and unit frequency
            pattern = self.catalog.get_unit(unit_text)
            freq_weight = math.log(pattern.count + 1) if pattern else 1.0
            weight = overlap_score * freq_weight

            for word, prob in right_dist.items():
                # Apply IDF to downweight common predicted words
                idf = self.word_idf.get(word, 1.0)
                contribution = weight * prob * idf
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

    def predict_units(self, tokens: List[str],
                      top_k: int = 10,
                      min_unit_length: int = 1,
                      max_unit_length: int = 5) -> List[UnitPrediction]:
        """
        Predict next units (multi-token sequences) rather than single tokens.

        This is analogous to beam search in transformers - we're finding
        coherent multi-token continuations rather than individual words.

        The key insight: A unit U can follow the prompt if U's left-context
        matches what typically follows the prompt's ending context.
        """
        if not tokens:
            return []

        # 1. Get the context signature at the end of the prompt
        # This tells us what kind of words/patterns typically follow this context
        context_sig = self.compute_context_signature(tokens, len(tokens))

        # 2. Find units whose left-context matches this signature
        # These are units that could plausibly follow the prompt
        candidate_units = []

        # Get the last few tokens as immediate context
        immediate_context = tokens[-min(3, len(tokens)):]
        immediate_set = set(immediate_context)

        for unit_text, pattern in self.catalog.units.items():
            if pattern.count < self.min_unit_freq:
                continue

            # Check unit length
            unit_tokens = unit_text.split()
            if len(unit_tokens) < min_unit_length or len(unit_tokens) > max_unit_length:
                continue

            # Score by how well unit's left-context matches our context
            unit_left = self.unit_left_contexts.get(unit_text, {})
            if not unit_left:
                continue

            # IDF-weighted similarity between prompt context and unit's left-context
            overlap_score = 0.0
            for word, query_weight in context_sig.left_words.items():
                if word in unit_left:
                    unit_weight = unit_left[word] * self.word_idf.get(word, 1.0)
                    overlap_score += query_weight * unit_weight

            # Bonus: if unit's left-context contains immediate context words
            immediate_bonus = 0.0
            for word in immediate_set:
                if word in unit_left:
                    immediate_bonus += unit_left[word] * 2.0

            total_score = overlap_score + immediate_bonus

            if total_score > 0.1:  # Threshold
                candidate_units.append((
                    unit_text,
                    unit_tokens,
                    total_score,
                    overlap_score,
                    pattern.count
                ))

        # 3. Sort by score and return top-k
        candidate_units.sort(key=lambda x: -x[2])

        predictions = []
        for unit_text, unit_tokens, score, overlap, freq in candidate_units[:top_k]:
            predictions.append(UnitPrediction(
                unit_text=unit_text,
                tokens=unit_tokens,
                score=score,
                context_overlap=overlap,
                frequency=freq
            ))

        return predictions

    def generate_with_units(self, prompt: List[str],
                            max_tokens: int = 20,
                            beam_width: int = 3,
                            length_penalty: float = 0.6,
                            no_repeat_ngram_size: int = 2) -> List[Tuple[List[str], float]]:
        """
        Generate text using unit-level beam search.

        Instead of predicting one token at a time, predict whole units
        and maintain multiple beams. Score sequences by cumulative
        context overlap, normalized by length.

        Args:
            no_repeat_ngram_size: Block n-grams of this size from repeating

        Returns: List of (generated_tokens, score) tuples for each beam.
        """
        def get_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
            """Extract all n-grams from a token list."""
            return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

        def would_repeat_ngram(existing: List[str], new_tokens: List[str], n: int) -> bool:
            """Check if adding new_tokens would create a repeated n-gram."""
            if n <= 0:
                return False
            # Check for consecutive word repetition (single token repeated)
            if existing and new_tokens:
                if existing[-1] == new_tokens[0]:
                    return True
                # Check if new_tokens start appears recently
                for i in range(1, min(4, len(existing) + 1)):
                    if existing[-i] == new_tokens[0]:
                        return True
            existing_ngrams = get_ngrams(existing, n)
            # Check n-grams that span the boundary or are in new_tokens
            combined = existing[-(n-1):] + new_tokens if len(existing) >= n-1 else existing + new_tokens
            new_ngrams = get_ngrams(combined, n)
            return bool(existing_ngrams & new_ngrams)

        # Initialize beams: (tokens, cumulative_score, num_units)
        beams = [(list(prompt), 0.0, 0)]

        for _ in range(max_tokens // 2):  # Rough iteration count
            new_beams = []

            for tokens, cum_score, num_units in beams:
                if len(tokens) - len(prompt) >= max_tokens:
                    new_beams.append((tokens, cum_score, num_units))
                    continue

                # Get unit predictions for this beam
                unit_preds = self.predict_units(tokens, top_k=beam_width * 4)

                if not unit_preds:
                    # No predictions - keep beam as is
                    new_beams.append((tokens, cum_score, num_units))
                    continue

                added = 0
                for pred in unit_preds:
                    if added >= beam_width:
                        break
                    # N-gram blocking: skip if this would repeat an n-gram
                    if would_repeat_ngram(tokens, pred.tokens, no_repeat_ngram_size):
                        continue
                    new_tokens = tokens + pred.tokens
                    new_score = cum_score + math.log(pred.score + 1e-10)
                    new_beams.append((new_tokens, new_score, num_units + 1))
                    added += 1

            # Prune to top beams, normalized by length
            def beam_score(beam):
                tokens, score, n_units = beam
                gen_len = len(tokens) - len(prompt)
                if gen_len == 0:
                    return float('-inf')
                # Length-normalized score (Wu et al. 2016 style)
                length_norm = ((5 + gen_len) / 6) ** length_penalty
                return score / length_norm

            new_beams.sort(key=beam_score, reverse=True)
            beams = new_beams[:beam_width]

            # Check if all beams have reached max length
            if all(len(b[0]) - len(prompt) >= max_tokens for b in beams):
                break

        # Return generated portions with scores
        results = []
        for tokens, score, n_units in beams:
            gen_tokens = tokens[len(prompt):]
            if gen_tokens:
                # Compute final normalized score
                length_norm = ((5 + len(gen_tokens)) / 6) ** length_penalty
                norm_score = score / length_norm if score != 0 else 0
                results.append((gen_tokens, norm_score))

        return results

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
                    # Apply IDF to downweight common predicted words
                    idf = self.word_idf.get(word, 1.0)
                    contribution = level_weight * freq_weight * prob * idf
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

    def find_substitution_class(self, unit_text: str,
                                 max_members: int = 30,
                                 threshold: float = 0.20) -> List[Tuple[str, float]]:
        """
        Find the substitution class for a unit using GPU-accelerated batch computation.

        The substitution class includes ALL units that share context patterns,
        regardless of length. This means:
        - "rain" can be in the class of "the rain in spain" (abstraction)
        - "the heavy rain" can be in the class of "rain" (elaboration)

        Returns: [(member_unit, centrality_score), ...]
        """
        # Check cache first
        if not hasattr(self, '_subclass_cache'):
            self._subclass_cache = {}
        if unit_text in self._subclass_cache:
            return self._subclass_cache[unit_text]

        # Check if unit exists
        if unit_text not in self.unit_to_idx:
            return []

        unit_idx = self.unit_to_idx[unit_text]

        # Get query's context indices
        if unit_idx not in self.left_context_topk or unit_idx not in self.right_context_topk:
            return []

        # Find candidates via context indices
        pattern = self.unit_patterns.get(unit_text)
        candidates = set()
        for word, _ in pattern.left_words.most_common(5):
            cands = self.left_context_index.get(word, set())
            candidates.update(list(cands)[:500])
        for word, _ in pattern.right_words.most_common(5):
            cands = self.right_context_index.get(word, set())
            candidates.update(list(cands)[:500])

        # Limit and filter candidates
        candidates.discard(unit_text)
        candidate_list = [c for c in list(candidates)[:2000]
                         if c in self.unit_to_idx]

        if not candidate_list:
            return []

        # When sharing GPU with catalog, use catalog's indices for GPU operations
        if hasattr(self, 'unit_to_gpu_idx') and self.unit_to_gpu_idx:
            # Shared GPU mode - translate to catalog indices
            if unit_text not in self.unit_to_gpu_idx:
                return []
            gpu_query_idx = self.unit_to_gpu_idx[unit_text]

            valid_gpu_indices = []
            valid_candidates = []
            for cand_text in candidate_list:
                cand_idx = self.unit_to_idx[cand_text]
                if cand_idx in self.left_context_topk and cand_idx in self.right_context_topk:
                    if cand_text in self.unit_to_gpu_idx:
                        valid_gpu_indices.append(self.unit_to_gpu_idx[cand_text])
                        valid_candidates.append(cand_text)

            if not valid_candidates:
                return []

            avg_sims = self._gpu_batch_similarity(gpu_query_idx, valid_gpu_indices)
        else:
            # Own GPU tensors - use our indices
            candidate_indices = [self.unit_to_idx[c] for c in candidate_list]

            # Filter to candidates that have both left and right contexts
            valid_indices = []
            valid_candidates = []
            for cand_idx, cand_text in zip(candidate_indices, candidate_list):
                if cand_idx in self.left_context_topk and cand_idx in self.right_context_topk:
                    valid_indices.append(cand_idx)
                    valid_candidates.append(cand_text)

            if not valid_candidates:
                return []

            # GPU-ACCELERATED: Batch compute similarities for all candidates
            avg_sims = self._gpu_batch_similarity(unit_idx, valid_indices)

        # Filter by threshold and sort
        mask = avg_sims >= threshold
        filtered_indices = np.where(mask)[0]

        class_members = [
            (valid_candidates[i], float(avg_sims[i]))
            for i in filtered_indices
        ]
        class_members.sort(key=lambda x: -x[1])
        result = class_members[:max_members]

        # Cache the result
        self._subclass_cache[unit_text] = result
        return result

    def predict_with_substitution_class(self, tokens: List[str],
                                         parser: SimpleBidirParser,
                                         top_k: int = 10,
                                         parallel: bool = True,
                                         n_workers: int = 4) -> List[UnitPrediction]:
        """
        Predict using full substitution class expansion at all hierarchy levels.

        This is the "attention-like" mechanism:
        1. Parse prompt into hierarchy
        2. At each node, find its substitution class (including shorter abstractions)
        3. Aggregate predictions from ALL class members at ALL levels

        The key insight: "rain" is in the substitution class of "the rain in spain"
        because they share context patterns. So predictions from "rain" (which are
        thematically coherent) flow to the full phrase.

        Args:
            parallel: If True, use parallel processing for substitution class computation
            n_workers: Number of parallel workers (default 4)
        """
        if len(tokens) < 2:
            return self.predict_units(tokens, top_k)

        # 1. Parse the prompt into a hierarchy
        tree = parser.parse(tokens)

        # 2. Collect substitution classes at each level (parallel or sequential)
        if parallel:
            all_class_members = self._collect_substitution_classes_parallel(tree, n_workers)
        else:
            all_class_members = []
            self._collect_substitution_classes_recursive(tree, all_class_members, level=0)

        # 3. Aggregate predictions from all class members
        prediction_scores = defaultdict(float)
        prediction_sources = defaultdict(list)

        for unit_text, level, centrality in all_class_members:
            right_dist = self.unit_right_contexts.get(unit_text, {})
            if not right_dist:
                continue

            # Weight by:
            # - Level: higher (more abstract) = captures broader theme
            # - Centrality: how central this member is to its class
            # - Unit frequency: reliability
            pattern = self.unit_patterns.get(unit_text)
            freq_weight = math.log(pattern.count + 1) if pattern else 1.0

            # Higher levels (leaves) get more weight as they're more specific
            # But also include abstract (shorter) class members
            level_weight = 1.0 + 0.3 * level

            for word, prob in right_dist.items():
                # IDF weight on predictions
                idf = self.word_idf.get(word, 1.0)
                contribution = level_weight * centrality * freq_weight * prob * idf
                prediction_scores[word] += contribution
                prediction_sources[word].append((f"L{level}:{unit_text}", contribution))

        # 4. Return top predictions as UnitPrediction objects
        if not prediction_scores:
            return self.predict_units(tokens, top_k)

        total = sum(prediction_scores.values())
        predictions = []

        for word, score in sorted(prediction_scores.items(), key=lambda x: -x[1])[:top_k]:
            # Find best unit prediction starting with this word
            unit_pred = self._find_unit_starting_with(word, tokens)
            if unit_pred:
                predictions.append(UnitPrediction(
                    unit_text=unit_pred[0],
                    tokens=unit_pred[0].split(),
                    score=score / total,
                    context_overlap=unit_pred[1],
                    frequency=unit_pred[2]
                ))
            else:
                # Fall back to single token
                predictions.append(UnitPrediction(
                    unit_text=word,
                    tokens=[word],
                    score=score / total,
                    context_overlap=0.0,
                    frequency=0
                ))

        return predictions

    def _collect_all_nodes(self, node: ParseNode, level: int = 0) -> List[Tuple[str, int]]:
        """Collect all nodes from parse tree as (unit_text, level) pairs."""
        nodes = [(node.span.text, level)]
        if node.left:
            nodes.extend(self._collect_all_nodes(node.left, level + 1))
        if node.right:
            nodes.extend(self._collect_all_nodes(node.right, level + 1))
        return nodes

    def _compute_node_substitution_class(self, args: Tuple[str, int]) -> List[Tuple[str, int, float]]:
        """Compute substitution class for a single node. Used for parallel processing."""
        unit_text, level = args
        result = []

        if unit_text in self.unit_patterns:
            result.append((unit_text, level, 1.0))
            sub_class = self.find_substitution_class(unit_text, max_members=20)
            for member_text, centrality in sub_class:
                result.append((member_text, level, centrality))

        return result

    def _collect_substitution_classes_parallel(self, tree: ParseNode,
                                                n_workers: int = 4) -> List[Tuple[str, int, float]]:
        """
        Collect substitution classes for all nodes in parallel.

        Uses ThreadPoolExecutor since the heavy lifting is in NumPy operations
        which release the GIL.
        """
        # Collect all nodes first
        nodes = self._collect_all_nodes(tree)

        # Process nodes in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(self._compute_node_substitution_class, nodes)
            for node_results in results:
                all_results.extend(node_results)

        return all_results

    def _collect_substitution_classes_recursive(self, node: ParseNode,
                                                  result: List[Tuple[str, int, float]],
                                                  level: int = 0):
        """
        Recursively collect substitution class members for each node in the parse tree.
        (Non-parallel version, kept for compatibility)

        For each node:
        1. Add the node's unit itself
        2. Find its substitution class (units with overlapping contexts)
        3. This naturally includes shorter abstractions like "rain" for "the rain in spain"
        """
        unit_text = node.span.text

        # Add the unit itself with full centrality
        if unit_text in self.unit_patterns:
            result.append((unit_text, level, 1.0))

            # Find substitution class for this node
            sub_class = self.find_substitution_class(unit_text, max_members=20)
            for member_text, centrality in sub_class:
                result.append((member_text, level, centrality))

        # Recurse into children
        if node.left:
            self._collect_substitution_classes_recursive(node.left, result, level + 1)
        if node.right:
            self._collect_substitution_classes_recursive(node.right, result, level + 1)

    def _find_unit_starting_with(self, word: str, context: List[str]) -> Optional[Tuple[str, float, int]]:
        """
        Find a multi-token unit that starts with the given word and fits the context.
        Returns: (unit_text, context_overlap, frequency) or None
        """
        # Find units starting with this word
        candidates = []
        for unit_text, pattern in self.unit_patterns.items():
            if unit_text.startswith(word + " ") or unit_text == word:
                # Score by context overlap
                context_counter = Counter(context[-5:])
                left_sim = context_similarity_aggregated(context_counter, pattern.left_words)
                if left_sim > 0.05:
                    candidates.append((unit_text, left_sim, pattern.count))

        if not candidates:
            return None

        # Return best match (highest overlap × frequency)
        candidates.sort(key=lambda x: -x[1] * math.log(x[2] + 1))
        return candidates[0]

    def generate_with_substitution(self, prompt: List[str],
                                    parser: SimpleBidirParser,
                                    max_tokens: int = 20,
                                    beam_width: int = 3) -> List[Tuple[List[str], float]]:
        """
        Generate text using iterative substitution class prediction.

        Each step:
        1. Predict best unit using full substitution class method
        2. Append unit to sequence
        3. Re-parse and repeat

        The "attention" comes from the substitution class at each step,
        which includes abstract representations that maintain thematic coherence.
        """
        beams = [(list(prompt), 0.0)]

        for _ in range(max_tokens // 2):
            new_beams = []

            for tokens, cum_score in beams:
                if len(tokens) - len(prompt) >= max_tokens:
                    new_beams.append((tokens, cum_score))
                    continue

                # Get predictions using substitution class method
                unit_preds = self.predict_with_substitution_class(tokens, parser, top_k=beam_width * 2)

                if not unit_preds:
                    new_beams.append((tokens, cum_score))
                    continue

                added = 0
                seen_starts = set()
                for pred in unit_preds:
                    if added >= beam_width:
                        break
                    # Avoid repetition
                    first_word = pred.tokens[0] if pred.tokens else ""
                    if first_word in seen_starts:
                        continue
                    if any(t in tokens[-4:] for t in pred.tokens[:2]):
                        continue

                    seen_starts.add(first_word)
                    new_tokens = tokens + pred.tokens
                    new_score = cum_score + math.log(pred.score + 1e-10)
                    new_beams.append((new_tokens, new_score))
                    added += 1

            if not new_beams:
                break

            # Prune beams
            new_beams.sort(key=lambda x: -x[1])
            beams = new_beams[:beam_width]

        # Return generated portions
        results = []
        for tokens, score in beams:
            gen_tokens = tokens[len(prompt):]
            if gen_tokens:
                results.append((gen_tokens, score))

        return results

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

    print("\n" + "=" * 80)
    print("UNIT-LEVEL PREDICTION (Multi-token sequences)")
    print("=" * 80)
    print("\nInstead of predicting single tokens, predict whole units.")
    print("This is analogous to beam search in transformers.")

    for prompt in test_prompts[:3]:  # Just first 3 for brevity
        print(f"\n{'-' * 60}")
        print(f"PROMPT: \"{' '.join(prompt)}\"")
        print('-' * 60)

        # Unit-level predictions
        unit_preds = predictor.predict_units(prompt, top_k=5, min_unit_length=2)
        print("\nTop unit predictions (multi-token):")
        for i, pred in enumerate(unit_preds, 1):
            print(f"  {i}. \"{pred.unit_text}\" (score: {pred.score:.3f}, freq: {pred.frequency})")

    print("\n" + "=" * 80)
    print("UNIT-LEVEL BEAM SEARCH GENERATION")
    print("=" * 80)

    test_prompts_gen = [
        ["i", "want", "to"],
        ["do", "you", "want", "to"],
        ["she", "said", "that"],
    ]

    for prompt in test_prompts_gen:
        print(f"\n{'-' * 60}")
        print(f"PROMPT: \"{' '.join(prompt)}\"")
        print('-' * 60)

        # Generate with unit-level beam search
        beams = predictor.generate_with_units(prompt, max_tokens=15, beam_width=3)
        print("\nBeam search results (unit-level):")
        for i, (gen_tokens, score) in enumerate(beams, 1):
            full_text = ' '.join(prompt + gen_tokens)
            print(f"  {i}. \"{full_text}\" (score: {score:.3f})")

    print("\n" + "=" * 80, flush=True)
    print("SUBSTITUTION CLASS PREDICTION (with 'attention'-like coherence)", flush=True)
    print("=" * 80, flush=True)
    print("\nKey insight: The substitution class of 'the rain in spain' includes 'rain'.", flush=True)
    print("Predictions from 'rain' (thematically coherent) flow to the full phrase.", flush=True)
    print("This is analogous to transformer attention over abstract representations.", flush=True)

    test_prompts_sub = [
        ["i", "want", "to"],
        ["she", "said", "that"],
        ["we", "should", "go"],
    ]

    for prompt in test_prompts_sub:
        print(f"\n{'-' * 60}", flush=True)
        print(f"PROMPT: \"{' '.join(prompt)}\"", flush=True)
        print('-' * 60, flush=True)

        # Show the substitution class for the full phrase
        full_phrase = ' '.join(prompt)
        print(f"\nSubstitution class for \"{full_phrase}\":", flush=True)
        sub_class = predictor.find_substitution_class(full_phrase, max_members=10)
        if sub_class:
            for member, centrality in sub_class[:5]:
                print(f"  - \"{member}\" (centrality: {centrality:.3f})", flush=True)
        else:
            print("  (phrase not in catalog, showing component classes)", flush=True)
            # Show classes for components
            for token in prompt[-2:]:
                tc = predictor.find_substitution_class(token, max_members=5)
                if tc:
                    print(f"  '{token}' class: {[m[0] for m in tc[:3]]}", flush=True)

        # Predictions using substitution class method
        print(f"\nComputing predictions (substitution class method)...", flush=True)
        preds = predictor.predict_with_substitution_class(prompt, parser, top_k=5)
        print(f"Predictions:", flush=True)
        for i, pred in enumerate(preds, 1):
            print(f"  {i}. \"{pred.unit_text}\" (score: {pred.score:.3f})", flush=True)

    print("\n" + "=" * 80)
    print("ITERATIVE GENERATION WITH SUBSTITUTION CLASSES")
    print("=" * 80)
    print("\nGenerate by iteratively predicting units using substitution class 'attention'.")

    for prompt in test_prompts_sub:
        print(f"\n{'-' * 60}")
        print(f"PROMPT: \"{' '.join(prompt)}\"")
        print('-' * 60)

        beams = predictor.generate_with_substitution(prompt, parser, max_tokens=15, beam_width=3)
        print("\nGenerated (substitution class method):")
        for i, (gen_tokens, score) in enumerate(beams, 1):
            full_text = ' '.join(prompt + gen_tokens)
            print(f"  {i}. \"{full_text}\" (score: {score:.3f})")


if __name__ == "__main__":
    main()
