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
from similarity import SimilarityMetrics, IDFCalculator


class BidirectionalExpander:
    """
    Performs bidirectional expansion of units and contexts.
    """

    def __init__(
        self,
        catalog: Catalog,
        similarity_threshold: float = 0.15,
        max_iterations: int = 3,
        similarity_metric: str = 'cosine_tfidf_min',
        verbose: bool = True
    ):
        """
        Args:
            catalog: The corpus catalog
            similarity_threshold: Minimum similarity to consider contexts related
            max_iterations: Maximum expansion iterations
            similarity_metric: Which metric to use for finding similar units
                Options: 'jaccard_avg', 'jaccard_min', 'jaccard_no_stop_avg',
                         'jaccard_no_stop_min', 'weighted_jaccard_avg',
                         'weighted_jaccard_min', 'cosine_tfidf_avg', 'cosine_tfidf_min'
            verbose: Print detailed progress information
        """
        self.catalog = catalog
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations
        self.similarity_metric = similarity_metric
        self.verbose = verbose

        # Initialize similarity metrics (computes IDF weights)
        self._log("Initializing similarity metrics...")
        self.sim_metrics = SimilarityMetrics(catalog, verbose=verbose)
        self._log(f"Using metric: {similarity_metric}, threshold: {similarity_threshold}")

    def _log(self, msg: str, indent: int = 0) -> None:
        """Print a message if verbose mode is on."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{msg}")

    def find_similar_units(self, unit: Unit, max_results: int = 100) -> Set[Unit]:
        """
        Find units with similar context distributions to the given unit.
        Uses the configured similarity metric.
        """
        similar_list = self.sim_metrics.find_similar_units(
            unit,
            metric=self.similarity_metric,
            threshold=self.similarity_threshold,
            max_results=max_results
        )
        return set(u for u, score in similar_list)

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
    import argparse
    from catalog import Catalog, build_catalog_from_cornell

    parser = argparse.ArgumentParser(description="Bidirectional expansion on Cornell Movie Dialogues")
    parser.add_argument("--max-sentences", type=int, default=10000,
                        help="Maximum sentences to load (default: 10000)")
    parser.add_argument("--context-size", type=int, default=3,
                        help="Context window size (default: 3)")
    parser.add_argument("--max-n", type=int, default=4,
                        help="Maximum n-gram size (default: 4)")
    parser.add_argument("--similarity-threshold", type=float, default=0.15,
                        help="Context similarity threshold (default: 0.15)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max expansion iterations (default: 3)")
    parser.add_argument("--similarity-metric", type=str, default='cosine_tfidf_min',
                        choices=['jaccard_avg', 'jaccard_min', 'jaccard_no_stop_avg',
                                 'jaccard_no_stop_min', 'weighted_jaccard_avg',
                                 'weighted_jaccard_min', 'cosine_tfidf_avg', 'cosine_tfidf_min'],
                        help="Similarity metric (default: cosine_tfidf_min)")
    args = parser.parse_args()

    print("="*70)
    print("BIDIRECTIONAL EXPANSION - Cornell Movie Dialogues Corpus")
    print("="*70)
    print(f"Parameters:")
    print(f"  max_sentences: {args.max_sentences}")
    print(f"  context_size: {args.context_size}")
    print(f"  max_n: {args.max_n}")
    print(f"  similarity_threshold: {args.similarity_threshold}")
    print(f"  max_iterations: {args.max_iterations}")
    print(f"  similarity_metric: {args.similarity_metric}")
    print()

    # Build catalog from Cornell corpus
    catalog = build_catalog_from_cornell(
        max_sentences=args.max_sentences,
        context_size=args.context_size,
        max_n=args.max_n,
        verbose=True
    )

    print("\n" + "="*70)
    print("CREATING EXPANDER")
    print("="*70)

    # Create expander
    expander = BidirectionalExpander(
        catalog,
        similarity_threshold=args.similarity_threshold,
        max_iterations=args.max_iterations,
        similarity_metric=args.similarity_metric,
        verbose=True
    )

    # Test expansions for different units likely to appear in movie dialogues
    test_units = [
        ("you", "know"),      # Very common filler
        ("i", "love"),        # Common in movies
        ("go", "out"),        # Social/motion polysemy
        ("my", "money"),      # Possessive NP
        ("want", "to"),       # Common verb phrase
        ("going", "to"),      # Future tense marker
        ("have", "to"),       # Modal-like
        ("get", "out"),       # Phrasal verb
    ]

    print(f"\nTesting {len(test_units)} units for expansion...")
    print("(Only testing units that exist in the catalog)")
    print()

    results = []
    for unit in test_units:
        freq = catalog.get_unit_frequency(unit)
        if freq == 0:
            print(f"\nSkipping '{' '.join(unit)}' - not found in catalog")
            continue

        print(f"\n{'#'*70}")
        print(f"TESTING UNIT: '{' '.join(unit)}' (frequency: {freq})")
        print(f"{'#'*70}")

        unit_exp, ctx_exp, energy = expander.expand(unit)
        results.append((unit, freq, len(unit_exp), len(ctx_exp), energy))

        # Show some sample units from expansion (not all, to keep output manageable)
        if len(unit_exp) > 20:
            print(f"\nSample of expanded units (showing 20 of {len(unit_exp)}):")
            sample_units = sorted(unit_exp, key=lambda u: -catalog.get_unit_frequency(u))[:20]
            for u in sample_units:
                print(f"  '{' '.join(u)}' (freq={catalog.get_unit_frequency(u)})")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON - Sorted by Energy (lower = better unit)")
    print("="*70)
    print(f"{'Unit':<20} {'Freq':<10} {'Units':<10} {'Contexts':<12} {'Energy':<10}")
    print("-"*62)
    for unit, freq, n_units, n_contexts, energy in sorted(results, key=lambda x: x[4]):
        unit_str = ' '.join(unit)
        print(f"{unit_str:<20} {freq:<10} {n_units:<10} {n_contexts:<12} {energy:<10.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Lower energy indicates a stronger syntactic unit:
- Large unit expansion = many substitutable phrases
- Large context expansion = diverse usage contexts
- Together = high substitutability across diverse contexts

Compare:
- True multi-word units (e.g., 'go out', 'get out') should have
  cohesive expansions with semantically related substitutes
- Accidental adjacencies should have smaller or less coherent expansions
""")
