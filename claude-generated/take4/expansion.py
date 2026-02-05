"""
Bidirectional Expansion Algorithm.

The core insight: simultaneously expand both the unit set (what can substitute)
and the context set (where they can substitute).

Score: E = -log(|unit_expansion| × log(|context_expansion| + 1))
Lower energy = larger expansion = better unit
"""

from collections import Counter
from typing import Set, Tuple, Dict, List, Optional
from catalog import Catalog, Unit, Context


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|"""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def counter_top_k_set(counter: Counter, k: int = 10) -> Set[str]:
    """Get the top-k most common items from a counter as a set."""
    return set(item for item, _ in counter.most_common(k))


def context_similarity(
    ctx1_left: Counter, ctx1_right: Counter,
    ctx2_left: Counter, ctx2_right: Counter,
    top_k: int = 10
) -> float:
    """
    Compute similarity between two contexts using their aggregated distributions.
    Uses Jaccard similarity on top-k words from each side.
    """
    # Get top-k words from each context's left and right distributions
    left1 = counter_top_k_set(ctx1_left, top_k)
    left2 = counter_top_k_set(ctx2_left, top_k)
    right1 = counter_top_k_set(ctx1_right, top_k)
    right2 = counter_top_k_set(ctx2_right, top_k)

    # Combine left and right similarity
    left_sim = jaccard_similarity(left1, left2)
    right_sim = jaccard_similarity(right1, right2)

    # Average of left and right similarity
    return (left_sim + right_sim) / 2


class BidirectionalExpander:
    """
    Performs bidirectional expansion of units and contexts.
    """

    def __init__(
        self,
        catalog: Catalog,
        similarity_threshold: float = 0.2,
        max_iterations: int = 3,
        top_k: int = 10,
        verbose: bool = True
    ):
        """
        Args:
            catalog: The corpus catalog
            similarity_threshold: Minimum similarity to consider contexts related
            max_iterations: Maximum expansion iterations
            top_k: Number of top words to use for similarity comparison
            verbose: Print detailed progress information
        """
        self.catalog = catalog
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.verbose = verbose

    def _log(self, msg: str, indent: int = 0) -> None:
        """Print a message if verbose mode is on."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{msg}")

    def find_similar_units(self, unit: Unit) -> Set[Unit]:
        """
        Find units with similar context distributions to the given unit.
        """
        similar = set()
        unit_left = self.catalog.get_left_distribution(unit)
        unit_right = self.catalog.get_right_distribution(unit)

        if not unit_left and not unit_right:
            return similar

        for other_unit in self.catalog.unit_to_contexts.keys():
            if other_unit == unit:
                continue

            other_left = self.catalog.get_left_distribution(other_unit)
            other_right = self.catalog.get_right_distribution(other_unit)

            sim = context_similarity(
                unit_left, unit_right,
                other_left, other_right,
                self.top_k
            )

            if sim >= self.similarity_threshold:
                similar.add(other_unit)

        return similar

    def find_units_for_context(self, context: Context) -> Set[Unit]:
        """
        Find all units that have appeared in a similar context.
        """
        # First, get exact matches
        exact_matches = self.catalog.get_units_in_context(context)

        # For now, just return exact matches
        # A more sophisticated version would also find units in similar contexts
        return exact_matches

    def expand(
        self,
        initial_unit: Unit,
        initial_context: Optional[Context] = None
    ) -> Tuple[Set[Unit], Set[Context], float]:
        """
        Perform bidirectional expansion starting from a unit and optional context.

        Returns:
            (unit_expansion, context_expansion, energy)
        """
        self._log(f"\n{'='*60}")
        self._log(f"BIDIRECTIONAL EXPANSION for unit: '{' '.join(initial_unit)}'")
        self._log(f"{'='*60}")

        # Initialize expansion sets
        unit_expansion: Set[Unit] = {initial_unit}
        context_expansion: Set[Context] = set()

        # If initial context provided, add it
        if initial_context:
            context_expansion.add(initial_context)
            self._log(f"Initial context: [{' '.join(initial_context[0])}] ___ [{' '.join(initial_context[1])}]")

        # Add all known contexts for the initial unit
        initial_contexts = self.catalog.get_contexts(initial_unit)
        context_expansion.update(initial_contexts)
        self._log(f"Initial unit frequency: {len(initial_contexts)}")
        self._log(f"Starting with {len(context_expansion)} context(s)")

        for iteration in range(self.max_iterations):
            self._log(f"\n--- Iteration {iteration + 1} ---", indent=1)

            prev_unit_count = len(unit_expansion)
            prev_context_count = len(context_expansion)

            # STEP 1: Expand units based on context similarity
            self._log("Step 1: Finding similar units...", indent=1)
            new_units = set()
            for unit in list(unit_expansion):
                similar = self.find_similar_units(unit)
                new_units.update(similar)

            added_units = new_units - unit_expansion
            unit_expansion.update(new_units)

            if added_units:
                self._log(f"Added {len(added_units)} new unit(s):", indent=2)
                for u in list(added_units)[:10]:  # Show first 10
                    self._log(f"+ '{' '.join(u)}'", indent=3)
                if len(added_units) > 10:
                    self._log(f"... and {len(added_units) - 10} more", indent=3)
            else:
                self._log("No new units found", indent=2)

            # STEP 2: Expand contexts based on new units
            self._log("Step 2: Gathering contexts from expanded units...", indent=1)
            new_contexts = set()
            for unit in unit_expansion:
                unit_contexts = self.catalog.get_contexts(unit)
                new_contexts.update(unit_contexts)

            added_contexts = new_contexts - context_expansion
            context_expansion.update(new_contexts)

            if added_contexts:
                self._log(f"Added {len(added_contexts)} new context(s):", indent=2)
                for ctx in list(added_contexts)[:5]:  # Show first 5
                    left, right = ctx
                    self._log(f"+ [{' '.join(left)}] ___ [{' '.join(right)}]", indent=3)
                if len(added_contexts) > 5:
                    self._log(f"... and {len(added_contexts) - 5} more", indent=3)
            else:
                self._log("No new contexts found", indent=2)

            # Check for convergence
            if len(unit_expansion) == prev_unit_count and len(context_expansion) == prev_context_count:
                self._log("Converged (no new units or contexts)", indent=1)
                break

            self._log(f"Expansion size: {len(unit_expansion)} units, {len(context_expansion)} contexts", indent=1)

        # Compute energy score
        import math
        if len(unit_expansion) > 0 and len(context_expansion) > 0:
            energy = -math.log(len(unit_expansion) * math.log(len(context_expansion) + 1))
        else:
            energy = float('inf')

        self._log(f"\n{'='*60}")
        self._log(f"FINAL EXPANSION RESULTS")
        self._log(f"{'='*60}")
        self._log(f"Unit expansion size: {len(unit_expansion)}")
        self._log(f"Context expansion size: {len(context_expansion)}")
        self._log(f"Energy: {energy:.4f}")

        if self.verbose:
            self._log(f"\nAll units in expansion:")
            for u in sorted(unit_expansion, key=lambda x: ' '.join(x)):
                freq = self.catalog.get_unit_frequency(u)
                self._log(f"  '{' '.join(u)}' (freq={freq})")

        return unit_expansion, context_expansion, energy


# --- Demo / Test ---

if __name__ == "__main__":
    from catalog import Catalog

    # Build catalog with sample sentences
    sample_sentences = [
        # Social "go out" contexts
        "Did you go out with Sarah",
        "Would you go out with me",
        "Let's go out on Friday",
        "We should hang out together",
        "Let's meet up on Friday",
        "Did you hang out with Helen",
        "Would you hang out with me",
        "We could get together sometime",
        "Did you meet up with John",

        # Motion "go out" contexts
        "I go out the door",
        "We go out the exit",
        "Please step out this way",
        "You should head out now",
        "I walk out the door",
        "We exit the building",

        # "my money" contexts (possessive NP)
        "I lost my money yesterday",
        "I need my money for rent",
        "Where is my money",
        "I found my keys yesterday",
        "I lost my keys yesterday",
        "She lost her wallet",
        "He found his phone",
        "I want your help",
        "I need your keys",
        "Where is your phone",
        "I lost the car",
        "I found the wallet",

        # Additional variety
        "The dog ran quickly",
        "A cat sat quietly",
        "The bird flew away",
    ]

    print("Building catalog...")
    catalog = Catalog(context_size=3, max_n=4)
    catalog.add_corpus(sample_sentences)
    print(f"Catalog: {catalog}")
    print()

    # Create expander
    expander = BidirectionalExpander(
        catalog,
        similarity_threshold=0.15,  # Lower threshold for small corpus
        max_iterations=3,
        top_k=5,  # Fewer top-k for small corpus
        verbose=True
    )

    # Test expansions for different units
    test_units = [
        ("go", "out"),
        ("my", "money"),
        ("lost", "my"),
        ("hang", "out"),
    ]

    results = []
    for unit in test_units:
        unit_exp, ctx_exp, energy = expander.expand(unit)
        results.append((unit, len(unit_exp), len(ctx_exp), energy))
        print("\n" + "="*80 + "\n")

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Unit':<20} {'Units':<10} {'Contexts':<10} {'Energy':<10}")
    print("-"*50)
    for unit, n_units, n_contexts, energy in sorted(results, key=lambda x: x[3]):
        unit_str = ' '.join(unit)
        print(f"{unit_str:<20} {n_units:<10} {n_contexts:<10} {energy:<10.4f}")
