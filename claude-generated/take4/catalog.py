"""
Catalog for bidirectional expansion parsing.

Stores:
- unit -> list of contexts (where this unit appeared)
- context -> set of units (what units appeared in this context)

A "context" is represented as a tuple: (left_tuple, right_tuple)
where left_tuple and right_tuple are tuples of tokens.
"""

from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional


# Type aliases for clarity
Token = str
Unit = Tuple[Token, ...]  # An n-gram represented as a tuple of tokens
Context = Tuple[Tuple[Token, ...], Tuple[Token, ...]]  # (left_context, right_context)


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


# --- Demo / Test ---

if __name__ == "__main__":
    # Sample sentences from the PDF examples
    sample_sentences = [
        "I lost my money yesterday",
        "I need my money for rent",
        "Where is my money",
        "I found your keys",
        "Did you go out with Sarah",
        "Would you go out with me",
        "Let's go out on Friday",
        "I go out the door",
        "We go out the exit",
        "I lost my keys yesterday",
        "She lost her wallet",
        "He found his phone",
        "I want your help",
        "We should hang out together",
        "Let's meet up on Friday",
        "Did you hang out with Helen",
    ]

    catalog = Catalog(context_size=3, max_n=4)
    catalog.add_corpus(sample_sentences)

    print("=== Catalog Stats ===")
    print(catalog)
    print()

    # Test some lookups
    test_units = [
        ("my", "money"),
        ("lost", "my"),
        ("go", "out"),
    ]

    for unit in test_units:
        print(f"=== Unit: {' '.join(unit)} ===")
        print(f"  Frequency: {catalog.get_unit_frequency(unit)}")
        contexts = catalog.get_contexts(unit)
        print(f"  Num contexts: {len(contexts)}")
        for ctx in contexts[:5]:  # Show first 5
            left, right = ctx
            print(f"    [{' '.join(left)}] ___ [{' '.join(right)}]")

        print(f"  Left dist (top 5): {catalog.get_left_distribution(unit).most_common(5)}")
        print(f"  Right dist (top 5): {catalog.get_right_distribution(unit).most_common(5)}")
        print()
