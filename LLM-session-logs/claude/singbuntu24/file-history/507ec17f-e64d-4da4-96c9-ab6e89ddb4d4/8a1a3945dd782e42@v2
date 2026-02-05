"""
Catalog for bidirectional expansion parsing.

Stores:
- unit -> list of contexts (where this unit appeared)
- context -> set of units (what units appeared in this context)

A "context" is represented as a tuple: (left_tuple, right_tuple)
where left_tuple and right_tuple are tuples of tokens.
"""

__version__ = "0.2.0"
__date__ = "2026-02-05"

import os
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional
from datetime import datetime
from pathlib import Path


# Type aliases for clarity
Token = str
Unit = Tuple[Token, ...]  # An n-gram represented as a tuple of tokens
Context = Tuple[Tuple[Token, ...], Tuple[Token, ...]]  # (left_context, right_context)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"


class OutputWriter:
    """Handles writing output to both console and file."""

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path
        self.file_handle = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_handle = open(output_path, 'w', encoding='utf-8')

    def write(self, msg: str = ""):
        print(msg)
        if self.file_handle:
            self.file_handle.write(msg + "\n")

    def close(self):
        if self.file_handle:
            self.file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Catalog:
    """
    A catalog that maps units to their contexts and vice versa.

    Built from a corpus of sentences, extracting n-grams and their
    surrounding context windows.
    """

    def __init__(self, context_size: int = 3, max_n: int = 4):
        """
        Args:
            context_size: Number of tokens to capture on each side of a unit
            max_n: Maximum n-gram size to extract
        """
        self.context_size = context_size
        self.max_n = max_n

        # unit -> list of all contexts where it appeared
        self.unit_to_contexts: Dict[Unit, List[Context]] = defaultdict(list)

        # context -> set of units that appeared in this context
        self.context_to_units: Dict[Context, Set[Unit]] = defaultdict(set)

        # unit -> frequency count
        self.unit_freq: Counter = Counter()

        # For aggregated context distributions (as mentioned in the PDF)
        # unit -> Counter of left context words
        self.unit_left_dist: Dict[Unit, Counter] = defaultdict(Counter)
        # unit -> Counter of right context words
        self.unit_right_dist: Dict[Unit, Counter] = defaultdict(Counter)

    def tokenize(self, sentence: str) -> List[Token]:
        """Simple whitespace tokenization with lowercasing."""
        return sentence.lower().split()

    def add_sentence(self, sentence: str) -> None:
        """
        Process a sentence and add all n-grams with their contexts to the catalog.
        """
        tokens = self.tokenize(sentence)
        n_tokens = len(tokens)

        # Extract n-grams of length 1 to max_n
        for n in range(1, self.max_n + 1):
            for i in range(n_tokens - n + 1):
                # The unit (n-gram)
                unit = tuple(tokens[i:i + n])

                # Left context: up to context_size tokens before the unit
                left_start = max(0, i - self.context_size)
                left_context = tuple(tokens[left_start:i])

                # Right context: up to context_size tokens after the unit
                right_end = min(n_tokens, i + n + self.context_size)
                right_context = tuple(tokens[i + n:right_end])

                context = (left_context, right_context)

                # Store in catalog
                self.unit_to_contexts[unit].append(context)
                self.context_to_units[context].add(unit)
                self.unit_freq[unit] += 1

                # Update aggregated distributions
                for tok in left_context:
                    self.unit_left_dist[unit][tok] += 1
                for tok in right_context:
                    self.unit_right_dist[unit][tok] += 1

    def add_corpus(self, sentences: List[str]) -> None:
        """Process multiple sentences."""
        for sentence in sentences:
            self.add_sentence(sentence)

    def get_contexts(self, unit: Unit) -> List[Context]:
        """Get all contexts where a unit appeared."""
        return self.unit_to_contexts.get(unit, [])

    def get_units_in_context(self, context: Context) -> Set[Unit]:
        """Get all units that appeared in a specific context."""
        return self.context_to_units.get(context, set())

    def get_unit_frequency(self, unit: Unit) -> int:
        """Get frequency count for a unit."""
        return self.unit_freq.get(unit, 0)

    def get_left_distribution(self, unit: Unit) -> Counter:
        """Get aggregated left context word distribution for a unit."""
        return self.unit_left_dist.get(unit, Counter())

    def get_right_distribution(self, unit: Unit) -> Counter:
        """Get aggregated right context word distribution for a unit."""
        return self.unit_right_dist.get(unit, Counter())

    def stats(self) -> Dict:
        """Return basic statistics about the catalog."""
        return {
            "num_unique_units": len(self.unit_to_contexts),
            "num_unique_contexts": len(self.context_to_units),
            "total_unit_occurrences": sum(self.unit_freq.values()),
            "most_common_units": self.unit_freq.most_common(10),
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"Catalog(units={stats['num_unique_units']}, "
            f"contexts={stats['num_unique_contexts']}, "
            f"occurrences={stats['total_unit_occurrences']})"
        )


def load_cornell_sentences(max_sentences: Optional[int] = None, verbose: bool = True,
                           writer: Optional[OutputWriter] = None) -> List[str]:
    """
    Load sentences from the Cornell Movie Dialogues Corpus via convokit.

    Args:
        max_sentences: Maximum number of sentences to load (None for all)
        verbose: Print progress information
        writer: OutputWriter for logging

    Returns:
        List of sentence strings
    """
    from convokit import Corpus, download

    def log(msg):
        if verbose:
            if writer:
                writer.write(msg)
            else:
                print(msg)

    log("Loading Cornell Movie Dialogues Corpus...")
    log("(This may download the corpus on first run)")

    corpus = Corpus(filename=download("movie-corpus"))

    sentences = []
    utterances = list(corpus.iter_utterances())
    total = len(utterances)

    log(f"Total utterances in corpus: {total}")

    for i, utterance in enumerate(utterances):
        text = utterance.text
        if text and len(text.strip()) > 0:
            # Clean up the text a bit
            text = text.strip()
            sentences.append(text)

        if max_sentences and len(sentences) >= max_sentences:
            break

        # Progress indicator
        if verbose and (i + 1) % 50000 == 0:
            log(f"  Processed {i + 1}/{total} utterances, collected {len(sentences)} sentences")

    log(f"Loaded {len(sentences)} sentences")

    return sentences


def build_catalog_from_cornell(
    max_sentences: Optional[int] = None,
    context_size: int = 3,
    max_n: int = 4,
    verbose: bool = True,
    writer: Optional[OutputWriter] = None
) -> "Catalog":
    """
    Build a catalog from the Cornell Movie Dialogues Corpus.

    Args:
        max_sentences: Maximum sentences to process (None for all)
        context_size: Context window size
        max_n: Maximum n-gram size
        verbose: Print progress
        writer: OutputWriter for logging

    Returns:
        Populated Catalog instance
    """
    def log(msg):
        if verbose:
            if writer:
                writer.write(msg)
            else:
                print(msg)

    sentences = load_cornell_sentences(max_sentences, verbose, writer)

    catalog = Catalog(context_size=context_size, max_n=max_n)

    log(f"\nBuilding catalog (context_size={context_size}, max_n={max_n})...")

    for i, sentence in enumerate(sentences):
        catalog.add_sentence(sentence)

        if verbose and (i + 1) % 10000 == 0:
            log(f"  Processed {i + 1}/{len(sentences)} sentences")

    log(f"Done. {catalog}")

    return catalog


# --- Demo / Test ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build catalog from Cornell Movie Dialogues")
    parser.add_argument("--max-sentences", type=int, default=10000,
                        help="Maximum sentences to load (default: 10000)")
    parser.add_argument("--context-size", type=int, default=3,
                        help="Context window size (default: 3)")
    parser.add_argument("--max-n", type=int, default=4,
                        help="Maximum n-gram size (default: 4)")
    parser.add_argument("--no-output-file", action="store_true",
                        help="Disable writing output to file")
    args = parser.parse_args()

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"catalog_{timestamp}.txt" if not args.no_output_file else None

    # Reconstruct command line
    cmd_line = "python " + " ".join(sys.argv)

    with OutputWriter(output_file) as writer:
        # Write header
        writer.write("=" * 70)
        writer.write("CATALOG BUILDER - Cornell Movie Dialogues Corpus")
        writer.write("=" * 70)
        writer.write(f"Version: {__version__} ({__date__})")
        writer.write(f"Run timestamp: {datetime.now().isoformat()}")
        writer.write(f"Command: {cmd_line}")
        writer.write("")
        writer.write("Parameters:")
        writer.write(f"  max_sentences: {args.max_sentences}")
        writer.write(f"  context_size: {args.context_size}")
        writer.write(f"  max_n: {args.max_n}")
        writer.write("")

        if output_file:
            writer.write(f"Output file: {output_file}")
            writer.write("")

        catalog = build_catalog_from_cornell(
            max_sentences=args.max_sentences,
            context_size=args.context_size,
            max_n=args.max_n,
            verbose=True,
            writer=writer
        )

        writer.write("")
        writer.write("=" * 70)
        writer.write("CATALOG STATISTICS")
        writer.write("=" * 70)
        stats = catalog.stats()
        writer.write(f"Unique units: {stats['num_unique_units']:,}")
        writer.write(f"Unique contexts: {stats['num_unique_contexts']:,}")
        writer.write(f"Total occurrences: {stats['total_unit_occurrences']:,}")
        writer.write("")
        writer.write("Most common units:")
        for unit, freq in stats['most_common_units']:
            writer.write(f"  '{' '.join(unit)}': {freq:,}")

        # Test some lookups
        writer.write("")
        writer.write("=" * 70)
        writer.write("SAMPLE UNIT LOOKUPS")
        writer.write("=" * 70)

        test_units = [
            ("my", "money"),
            ("go", "out"),
            ("i", "love"),
            ("you", "know"),
            ("want", "to"),
        ]

        for unit in test_units:
            freq = catalog.get_unit_frequency(unit)
            if freq == 0:
                writer.write(f"\n=== Unit: '{' '.join(unit)}' === NOT FOUND")
                continue

            writer.write(f"\n=== Unit: '{' '.join(unit)}' ===")
            writer.write(f"  Frequency: {freq:,}")

            contexts = catalog.get_contexts(unit)
            writer.write(f"  Unique context instances: {len(contexts)}")
            writer.write("  Sample contexts:")
            for ctx in contexts[:5]:
                left, right = ctx
                writer.write(f"    [{' '.join(left)}] ___ [{' '.join(right)}]")

            left_dist = catalog.get_left_distribution(unit)
            right_dist = catalog.get_right_distribution(unit)
            writer.write(f"  Left dist (top 5): {left_dist.most_common(5)}")
            writer.write(f"  Right dist (top 5): {right_dist.most_common(5)}")

        if output_file:
            writer.write("")
            writer.write("=" * 70)
            writer.write(f"Output saved to: {output_file}")
            writer.write("=" * 70)
