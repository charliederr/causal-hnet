"""
Top-Down Parser via Bidirectional Context Expansion.

The key insight: good syntactic units have large bidirectional expansions.
We parse by recursively finding split points that maximize the "unithood"
of the resulting constituents.

Algorithm:
1. For the full sentence span, enumerate all possible binary splits
2. For each split, compute the combined energy of the two children
3. Choose the split that produces children with lowest combined energy
4. Recursively parse each child
5. Stop when a span is a single token or has no good split
"""

__version__ = "0.2.0"
__date__ = "2026-02-05"

import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from datetime import datetime
from pathlib import Path

from catalog import Catalog, Unit, Context, OutputWriter, build_catalog_from_cornell
from expansion import BidirectionalExpander


# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class ParseNode:
    """A node in the parse tree."""
    tokens: Tuple[str, ...]
    start: int
    end: int
    energy: float
    children: Optional[Tuple['ParseNode', 'ParseNode']] = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def span_str(self) -> str:
        return ' '.join(self.tokens)

    def to_bracket_string(self, depth: int = 0) -> str:
        """Convert to bracketed string representation."""
        if self.is_leaf:
            return f"[{self.span_str}]"
        else:
            left_str = self.children[0].to_bracket_string(depth + 1)
            right_str = self.children[1].to_bracket_string(depth + 1)
            return f"[{left_str} {right_str}]"

    def to_tree_string(self, indent: int = 0) -> str:
        """Convert to indented tree representation."""
        prefix = "  " * indent
        energy_str = f"{self.energy:.3f}" if self.energy != float('inf') else "inf"
        result = f"{prefix}({self.span_str}) E={energy_str}\n"
        if not self.is_leaf:
            result += self.children[0].to_tree_string(indent + 1)
            result += self.children[1].to_tree_string(indent + 1)
        return result


class SpanEnergyCache:
    """
    Cache for span energies to avoid recomputation.

    Since full bidirectional expansion is expensive, we cache results
    and also provide a "quick" mode that uses simpler heuristics.
    """

    def __init__(
        self,
        catalog: Catalog,
        expander: Optional[BidirectionalExpander] = None,
        use_full_expansion: bool = False,
        min_frequency: int = 2,
        verbose: bool = True,
        writer: Optional[OutputWriter] = None
    ):
        """
        Args:
            catalog: The corpus catalog
            expander: Optional expander for full expansion (expensive)
            use_full_expansion: Whether to use full expansion (slow) or quick heuristics
            min_frequency: Minimum frequency for a span to be considered a valid unit
            verbose: Print progress
            writer: OutputWriter for logging
        """
        self.catalog = catalog
        self.expander = expander
        self.use_full_expansion = use_full_expansion
        self.min_frequency = min_frequency
        self.verbose = verbose
        self.writer = writer

        # Cache: unit tuple -> energy
        self._cache: Dict[Unit, float] = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _log(self, msg: str, indent: int = 0) -> None:
        if self.verbose:
            prefix = "  " * indent
            full_msg = f"{prefix}{msg}"
            if self.writer:
                self.writer.write(full_msg)
            else:
                print(full_msg)

    def get_energy(self, unit: Unit) -> float:
        """
        Get the energy for a unit (span).

        Lower energy = better unit (larger expansion potential).
        """
        if unit in self._cache:
            self.cache_hits += 1
            return self._cache[unit]

        self.cache_misses += 1

        # Check if unit exists in catalog with sufficient frequency
        freq = self.catalog.get_unit_frequency(unit)

        if freq < self.min_frequency:
            # Unknown or rare unit - assign high energy (bad unit)
            energy = float('inf')
        elif self.use_full_expansion and self.expander:
            # Full expansion (expensive but accurate)
            _, _, energy = self.expander.expand(unit)
        else:
            # Quick heuristic: use frequency and context diversity
            energy = self._quick_energy(unit, freq)

        self._cache[unit] = energy
        return energy

    def _quick_energy(self, unit: Unit, freq: int) -> float:
        """
        Quick energy estimate without full expansion.

        Heuristic: units that appear in diverse contexts are better.
        We estimate this using:
        - Frequency (more occurrences = more potential contexts)
        - Context diversity (number of unique contexts)
        """
        contexts = self.catalog.get_contexts(unit)
        n_unique_contexts = len(set(contexts))

        if n_unique_contexts == 0:
            return float('inf')

        # Simple energy: -log(freq * log(n_contexts + 1))
        # This mirrors the full expansion formula but uses raw counts
        try:
            energy = -math.log(freq * math.log(n_unique_contexts + 1))
        except (ValueError, ZeroDivisionError):
            energy = float('inf')

        return energy

    def precompute_for_sentence(self, tokens: List[str], max_span_length: int = 6) -> None:
        """
        Precompute energies for all spans in a sentence up to max length.
        This speeds up parsing by doing batch computation.
        """
        n = len(tokens)
        total_spans = 0
        computed = 0

        for length in range(1, min(max_span_length + 1, n + 1)):
            for start in range(n - length + 1):
                unit = tuple(tokens[start:start + length])
                if unit not in self._cache:
                    self.get_energy(unit)
                    computed += 1
                total_spans += 1

        self._log(f"Precomputed {computed} new span energies ({total_spans} total spans)")


class TopDownParser:
    """
    Top-down constituency parser using bidirectional expansion energy.

    The parser recursively splits spans at the point that produces
    children with the lowest combined energy (best units).
    """

    def __init__(
        self,
        catalog: Catalog,
        expander: Optional[BidirectionalExpander] = None,
        use_full_expansion: bool = False,
        min_frequency: int = 2,
        min_span_length: int = 1,
        max_span_length: int = 10,
        split_criterion: str = 'sum',
        verbose: bool = True,
        writer: Optional[OutputWriter] = None
    ):
        """
        Args:
            catalog: The corpus catalog
            expander: Optional expander for full expansion
            use_full_expansion: Use full expansion (slow) vs quick heuristics
            min_frequency: Minimum frequency for valid units
            min_span_length: Minimum span length to consider splitting
            max_span_length: Maximum span length for energy computation
            split_criterion: How to combine child energies ('sum', 'max', 'product')
            verbose: Print progress
            writer: OutputWriter for logging
        """
        self.catalog = catalog
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.split_criterion = split_criterion
        self.verbose = verbose
        self.writer = writer

        # Initialize energy cache
        self.energy_cache = SpanEnergyCache(
            catalog=catalog,
            expander=expander,
            use_full_expansion=use_full_expansion,
            min_frequency=min_frequency,
            verbose=verbose,
            writer=writer
        )

    def _log(self, msg: str, indent: int = 0) -> None:
        if self.verbose:
            prefix = "  " * indent
            full_msg = f"{prefix}{msg}"
            if self.writer:
                self.writer.write(full_msg)
            else:
                print(full_msg)

    def _combine_energies(self, e1: float, e2: float) -> float:
        """Combine two child energies based on the criterion."""
        if e1 == float('inf') or e2 == float('inf'):
            return float('inf')

        if self.split_criterion == 'sum':
            return e1 + e2
        elif self.split_criterion == 'max':
            return max(e1, e2)
        elif self.split_criterion == 'product':
            return e1 * e2
        else:
            return e1 + e2  # default to sum

    def _find_best_split(
        self,
        tokens: List[str],
        start: int,
        end: int,
        depth: int = 0
    ) -> Optional[int]:
        """
        Find the best split point for a span.

        Returns the split index k such that [start:k] and [k:end]
        have the lowest combined energy, or None if no good split exists.
        """
        span_length = end - start

        # Don't split very short spans
        if span_length <= self.min_span_length:
            return None

        best_split = None
        best_score = float('inf')

        # Try all possible split points
        for k in range(start + 1, end):
            left_unit = tuple(tokens[start:k])
            right_unit = tuple(tokens[k:end])

            # Skip if either child is too long for energy computation
            if len(left_unit) > self.max_span_length or len(right_unit) > self.max_span_length:
                # Use a penalty based on length
                left_energy = float('inf') if len(left_unit) > self.max_span_length else self.energy_cache.get_energy(left_unit)
                right_energy = float('inf') if len(right_unit) > self.max_span_length else self.energy_cache.get_energy(right_unit)
            else:
                left_energy = self.energy_cache.get_energy(left_unit)
                right_energy = self.energy_cache.get_energy(right_unit)

            combined = self._combine_energies(left_energy, right_energy)

            if combined < best_score:
                best_score = combined
                best_split = k

        # Only return split if it produces finite-energy children
        if best_score == float('inf'):
            return None

        return best_split

    def _parse_recursive(
        self,
        tokens: List[str],
        start: int,
        end: int,
        depth: int = 0
    ) -> ParseNode:
        """
        Recursively parse a span.
        """
        span_tokens = tuple(tokens[start:end])
        span_length = end - start

        # Get energy for this span
        if span_length <= self.max_span_length:
            energy = self.energy_cache.get_energy(span_tokens)
        else:
            energy = float('inf')

        self._log(f"Parsing [{start}:{end}] '{' '.join(span_tokens)}' (E={energy:.3f})", indent=depth)

        # Base case: single token or minimum span
        if span_length <= self.min_span_length:
            return ParseNode(
                tokens=span_tokens,
                start=start,
                end=end,
                energy=energy,
                children=None
            )

        # Find best split
        split_point = self._find_best_split(tokens, start, end, depth)

        if split_point is None:
            # No good split found - treat as leaf
            self._log(f"No good split found, treating as leaf", indent=depth + 1)
            return ParseNode(
                tokens=span_tokens,
                start=start,
                end=end,
                energy=energy,
                children=None
            )

        # Recursively parse children
        self._log(f"Splitting at position {split_point}", indent=depth + 1)

        left_child = self._parse_recursive(tokens, start, split_point, depth + 1)
        right_child = self._parse_recursive(tokens, split_point, end, depth + 1)

        return ParseNode(
            tokens=span_tokens,
            start=start,
            end=end,
            energy=energy,
            children=(left_child, right_child)
        )

    def parse(self, sentence: str, precompute: bool = True) -> ParseNode:
        """
        Parse a sentence and return a parse tree.

        Args:
            sentence: The sentence to parse
            precompute: Whether to precompute span energies first

        Returns:
            Root ParseNode of the parse tree
        """
        # Tokenize
        tokens = sentence.lower().split()

        self._log(f"\n{'='*60}")
        self._log(f"PARSING: {sentence}")
        self._log(f"Tokens: {tokens}")
        self._log(f"{'='*60}")

        if len(tokens) == 0:
            raise ValueError("Empty sentence")

        # Optionally precompute energies for all spans
        if precompute:
            self._log(f"\nPrecomputing span energies...")
            self.energy_cache.precompute_for_sentence(tokens, self.max_span_length)

        self._log(f"\nRecursive parsing:")

        # Parse the full span
        root = self._parse_recursive(tokens, 0, len(tokens), 0)

        self._log(f"\n{'='*60}")
        self._log(f"PARSE RESULT")
        self._log(f"{'='*60}")
        self._log(f"Bracketed: {root.to_bracket_string()}")
        self._log(f"\nTree:")
        self._log(root.to_tree_string())

        return root

    def parse_batch(self, sentences: List[str]) -> List[ParseNode]:
        """Parse multiple sentences."""
        results = []
        for i, sentence in enumerate(sentences):
            self._log(f"\n{'#'*60}")
            self._log(f"Sentence {i+1}/{len(sentences)}")
            self._log(f"{'#'*60}")
            results.append(self.parse(sentence))
        return results


# --- Demo / Test ---

if __name__ == "__main__":
    import argparse

    parser_args = argparse.ArgumentParser(description="Top-down parser using bidirectional expansion")
    parser_args.add_argument("--max-sentences", type=int, default=10000,
                        help="Maximum sentences to load for catalog (default: 10000)")
    parser_args.add_argument("--context-size", type=int, default=3,
                        help="Context window size (default: 3)")
    parser_args.add_argument("--max-n", type=int, default=4,
                        help="Maximum n-gram size for catalog (default: 4)")
    parser_args.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum frequency for valid units (default: 2)")
    parser_args.add_argument("--min-span-length", type=int, default=1,
                        help="Minimum span length to split (default: 1)")
    parser_args.add_argument("--max-span-length", type=int, default=6,
                        help="Maximum span length for energy computation (default: 6)")
    parser_args.add_argument("--split-criterion", type=str, default='sum',
                        choices=['sum', 'max', 'product'],
                        help="How to combine child energies (default: sum)")
    parser_args.add_argument("--use-full-expansion", action="store_true",
                        help="Use full bidirectional expansion (slow but more accurate)")
    parser_args.add_argument("--similarity-threshold", type=float, default=0.15,
                        help="Similarity threshold for expansion (default: 0.15)")
    parser_args.add_argument("--similarity-metric", type=str, default='cosine_tfidf_min',
                        help="Similarity metric for expansion (default: cosine_tfidf_min)")
    parser_args.add_argument("--sentence", type=str, default=None,
                        help="Single sentence to parse (optional)")
    parser_args.add_argument("--no-output-file", action="store_true",
                        help="Disable writing output to file")
    args = parser_args.parse_args()

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"parser_{timestamp}.txt" if not args.no_output_file else None

    # Reconstruct command line
    cmd_line = "python " + " ".join(sys.argv)

    with OutputWriter(output_file) as writer:
        # Write header
        writer.write("=" * 70)
        writer.write("TOP-DOWN PARSER - Bidirectional Expansion")
        writer.write("=" * 70)
        writer.write(f"Version: {__version__} ({__date__})")
        writer.write(f"Run timestamp: {datetime.now().isoformat()}")
        writer.write(f"Command: {cmd_line}")
        writer.write("")
        writer.write("Parameters:")
        writer.write(f"  max_sentences: {args.max_sentences}")
        writer.write(f"  context_size: {args.context_size}")
        writer.write(f"  max_n: {args.max_n}")
        writer.write(f"  min_frequency: {args.min_frequency}")
        writer.write(f"  min_span_length: {args.min_span_length}")
        writer.write(f"  max_span_length: {args.max_span_length}")
        writer.write(f"  split_criterion: {args.split_criterion}")
        writer.write(f"  use_full_expansion: {args.use_full_expansion}")
        if args.use_full_expansion:
            writer.write(f"  similarity_threshold: {args.similarity_threshold}")
            writer.write(f"  similarity_metric: {args.similarity_metric}")
        writer.write("")

        if output_file:
            writer.write(f"Output file: {output_file}")
            writer.write("")

        # Build catalog
        catalog = build_catalog_from_cornell(
            max_sentences=args.max_sentences,
            context_size=args.context_size,
            max_n=args.max_n,
            verbose=True,
            writer=writer
        )

        writer.write("")
        writer.write("=" * 70)
        writer.write("CREATING PARSER")
        writer.write("=" * 70)

        # Optionally create expander for full expansion mode
        expander = None
        if args.use_full_expansion:
            from expansion import BidirectionalExpander
            expander = BidirectionalExpander(
                catalog,
                similarity_threshold=args.similarity_threshold,
                similarity_metric=args.similarity_metric,
                max_iterations=2,
                verbose=False,  # Don't spam during parsing
                writer=None
            )

        # Create parser
        td_parser = TopDownParser(
            catalog=catalog,
            expander=expander,
            use_full_expansion=args.use_full_expansion,
            min_frequency=args.min_frequency,
            min_span_length=args.min_span_length,
            max_span_length=args.max_span_length,
            split_criterion=args.split_criterion,
            verbose=True,
            writer=writer
        )

        # Test sentences
        if args.sentence:
            test_sentences = [args.sentence]
        else:
            test_sentences = [
                "I want to go out with you",
                "She gave him the book",
                "The cat sat on the mat",
                "I don't know what to do",
                "Do you want to go out",
                "I love you",
                "What do you think",
                "Let me know if you need help",
            ]

        writer.write(f"\nParsing {len(test_sentences)} test sentence(s)...")
        writer.write("")

        # Parse each sentence
        results = []
        for sentence in test_sentences:
            try:
                tree = td_parser.parse(sentence)
                results.append((sentence, tree))
            except Exception as e:
                writer.write(f"Error parsing '{sentence}': {e}")
                results.append((sentence, None))

        # Summary
        writer.write("")
        writer.write("=" * 70)
        writer.write("SUMMARY")
        writer.write("=" * 70)

        for sentence, tree in results:
            if tree:
                writer.write(f"\n'{sentence}'")
                writer.write(f"  -> {tree.to_bracket_string()}")
            else:
                writer.write(f"\n'{sentence}'")
                writer.write(f"  -> PARSE FAILED")

        writer.write("")
        writer.write(f"\nCache statistics:")
        writer.write(f"  Hits: {td_parser.energy_cache.cache_hits}")
        writer.write(f"  Misses: {td_parser.energy_cache.cache_misses}")

        if output_file:
            writer.write("")
            writer.write("=" * 70)
            writer.write(f"Output saved to: {output_file}")
            writer.write("=" * 70)
