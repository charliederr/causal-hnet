"""
Context Similarity Metrics for Bidirectional Expansion.

The problem with basic Jaccard on top-k words:
  "go out" context: (left: [to, you, i], right: [with, and, the])
  "a big" context:  (left: [to, was, i], right: [and, the, for])
  Jaccard = 0.4 (overlaps on: to, i, and, the)
  => Incorrectly deemed similar!

This module explores alternative metrics:
1. IDF-weighted similarity (downweight common words)
2. Stopword filtering
3. PMI-based weighting
4. Requiring BOTH left and right to match (not average)
5. Cosine similarity on TF-IDF vectors
"""

import math
from collections import Counter
from typing import Set, Dict, List, Tuple, Optional
from catalog import Catalog, Unit


# Common English stopwords that inflate false similarity
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    # Contractions common in movie dialogue
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
    "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
    "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll",
    "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot",
    "couldn't", "mustn't", "let's", "that's", "who's", "what's",
    "here's", "there's", "when's", "where's", "why's", "how's",
}


class IDFCalculator:
    """
    Compute IDF (Inverse Document Frequency) weights for context words.

    IDF(word) = log(N / df(word))
    where N = total units, df(word) = number of units with this word in context
    """

    def __init__(self, catalog: Catalog, verbose: bool = True):
        self.catalog = catalog
        self.verbose = verbose

        # Count document frequency for each word
        # A "document" here is a unit's aggregated context
        self.left_df: Counter = Counter()  # word -> num units with this word in left context
        self.right_df: Counter = Counter()  # word -> num units with this word in right context
        self.total_units = len(catalog.unit_to_contexts)

        self._compute_df()

    def _compute_df(self):
        """Compute document frequencies."""
        if self.verbose:
            print(f"Computing IDF weights for {self.total_units} units...")

        for unit in self.catalog.unit_to_contexts.keys():
            left_dist = self.catalog.get_left_distribution(unit)
            right_dist = self.catalog.get_right_distribution(unit)

            # Each unique word in this unit's context counts once
            for word in left_dist.keys():
                self.left_df[word] += 1
            for word in right_dist.keys():
                self.right_df[word] += 1

        if self.verbose:
            print(f"  Left context vocabulary: {len(self.left_df)} words")
            print(f"  Right context vocabulary: {len(self.right_df)} words")

            # Show most common (lowest IDF) words
            print(f"  Most common left context words (low IDF):")
            for word, df in self.left_df.most_common(10):
                idf = math.log(self.total_units / df) if df > 0 else 0
                print(f"    '{word}': df={df}, idf={idf:.3f}")

            print(f"  Most common right context words (low IDF):")
            for word, df in self.right_df.most_common(10):
                idf = math.log(self.total_units / df) if df > 0 else 0
                print(f"    '{word}': df={df}, idf={idf:.3f}")

    def get_left_idf(self, word: str) -> float:
        """Get IDF weight for a word in left context."""
        df = self.left_df.get(word, 0)
        if df == 0:
            return 0.0
        return math.log(self.total_units / df)

    def get_right_idf(self, word: str) -> float:
        """Get IDF weight for a word in right context."""
        df = self.right_df.get(word, 0)
        if df == 0:
            return 0.0
        return math.log(self.total_units / df)


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Basic Jaccard: |A ∩ B| / |A ∪ B|"""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def jaccard_no_stopwords(counter1: Counter, counter2: Counter, top_k: int = 10) -> float:
    """Jaccard similarity after removing stopwords."""
    # Get top-k non-stopword items
    def top_k_filtered(counter: Counter) -> Set[str]:
        result = set()
        for word, _ in counter.most_common():
            if word.lower() not in STOPWORDS:
                result.add(word)
                if len(result) >= top_k:
                    break
        return result

    set1 = top_k_filtered(counter1)
    set2 = top_k_filtered(counter2)

    return jaccard_similarity(set1, set2)


def weighted_jaccard(
    counter1: Counter,
    counter2: Counter,
    idf_func,
    top_k: int = 10
) -> float:
    """
    Weighted Jaccard using IDF weights.

    weighted_intersection = sum(min(w1[i], w2[i]) * idf[i])
    weighted_union = sum(max(w1[i], w2[i]) * idf[i])
    """
    # Get vocabulary from both
    all_words = set(counter1.keys()) | set(counter2.keys())

    # Filter to top-k by IDF-weighted frequency from each
    def idf_weighted_top_k(counter: Counter) -> Set[str]:
        scored = [(word, count * idf_func(word)) for word, count in counter.items()]
        scored.sort(key=lambda x: -x[1])
        return set(word for word, _ in scored[:top_k])

    words1 = idf_weighted_top_k(counter1)
    words2 = idf_weighted_top_k(counter2)
    relevant_words = words1 | words2

    if not relevant_words:
        return 0.0

    weighted_intersection = 0.0
    weighted_union = 0.0

    for word in relevant_words:
        w1 = counter1.get(word, 0)
        w2 = counter2.get(word, 0)
        idf = idf_func(word)

        weighted_intersection += min(w1, w2) * idf
        weighted_union += max(w1, w2) * idf

    return weighted_intersection / weighted_union if weighted_union > 0 else 0.0


def cosine_similarity_tfidf(
    counter1: Counter,
    counter2: Counter,
    idf_func,
    top_k: int = 20
) -> float:
    """
    Cosine similarity using TF-IDF vectors.

    TF = count / total_count (normalized term frequency)
    TF-IDF = TF * IDF
    """
    # Build TF-IDF vectors
    def build_tfidf(counter: Counter) -> Dict[str, float]:
        total = sum(counter.values())
        if total == 0:
            return {}
        return {word: (count / total) * idf_func(word)
                for word, count in counter.items()}

    vec1 = build_tfidf(counter1)
    vec2 = build_tfidf(counter2)

    # Get top-k by TF-IDF value from each
    def top_k_words(vec: Dict[str, float]) -> Set[str]:
        items = sorted(vec.items(), key=lambda x: -x[1])
        return set(word for word, _ in items[:top_k])

    words1 = top_k_words(vec1)
    words2 = top_k_words(vec2)
    all_words = words1 | words2

    if not all_words:
        return 0.0

    # Compute cosine similarity
    dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in all_words)
    norm1 = math.sqrt(sum(vec1.get(w, 0)**2 for w in all_words))
    norm2 = math.sqrt(sum(vec2.get(w, 0)**2 for w in all_words))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SimilarityMetrics:
    """
    Collection of similarity metrics for comparing unit contexts.
    """

    def __init__(self, catalog: Catalog, verbose: bool = True):
        self.catalog = catalog
        self.verbose = verbose
        self.idf = IDFCalculator(catalog, verbose)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def compare_unit_pair(
        self,
        unit1: Unit,
        unit2: Unit,
        show_details: bool = True
    ) -> Dict[str, float]:
        """
        Compare two units using all available similarity metrics.
        Returns dict of metric_name -> score.
        """
        left1 = self.catalog.get_left_distribution(unit1)
        right1 = self.catalog.get_right_distribution(unit1)
        left2 = self.catalog.get_left_distribution(unit2)
        right2 = self.catalog.get_right_distribution(unit2)

        results = {}

        # 1. Basic Jaccard on top-10
        def basic_jaccard(c1, c2, k=10):
            s1 = set(w for w, _ in c1.most_common(k))
            s2 = set(w for w, _ in c2.most_common(k))
            return jaccard_similarity(s1, s2)

        left_jac = basic_jaccard(left1, left2)
        right_jac = basic_jaccard(right1, right2)
        results['jaccard_avg'] = (left_jac + right_jac) / 2
        results['jaccard_min'] = min(left_jac, right_jac)
        results['jaccard_left'] = left_jac
        results['jaccard_right'] = right_jac

        # 2. Jaccard without stopwords
        left_jac_ns = jaccard_no_stopwords(left1, left2)
        right_jac_ns = jaccard_no_stopwords(right1, right2)
        results['jaccard_no_stop_avg'] = (left_jac_ns + right_jac_ns) / 2
        results['jaccard_no_stop_min'] = min(left_jac_ns, right_jac_ns)

        # 3. IDF-weighted Jaccard
        left_wjac = weighted_jaccard(left1, left2, self.idf.get_left_idf)
        right_wjac = weighted_jaccard(right1, right2, self.idf.get_right_idf)
        results['weighted_jaccard_avg'] = (left_wjac + right_wjac) / 2
        results['weighted_jaccard_min'] = min(left_wjac, right_wjac)

        # 4. Cosine TF-IDF
        left_cos = cosine_similarity_tfidf(left1, left2, self.idf.get_left_idf)
        right_cos = cosine_similarity_tfidf(right1, right2, self.idf.get_right_idf)
        results['cosine_tfidf_avg'] = (left_cos + right_cos) / 2
        results['cosine_tfidf_min'] = min(left_cos, right_cos)

        if show_details:
            u1_str = ' '.join(unit1)
            u2_str = ' '.join(unit2)
            self._log(f"\n{'='*60}")
            self._log(f"Comparing: '{u1_str}' vs '{u2_str}'")
            self._log(f"{'='*60}")

            self._log(f"\n'{u1_str}' contexts:")
            self._log(f"  Left (top 10):  {left1.most_common(10)}")
            self._log(f"  Right (top 10): {right1.most_common(10)}")

            self._log(f"\n'{u2_str}' contexts:")
            self._log(f"  Left (top 10):  {left2.most_common(10)}")
            self._log(f"  Right (top 10): {right2.most_common(10)}")

            self._log(f"\nSimilarity scores:")
            for name, score in sorted(results.items()):
                self._log(f"  {name}: {score:.4f}")

        return results

    def find_similar_units(
        self,
        unit: Unit,
        metric: str = 'cosine_tfidf_min',
        threshold: float = 0.1,
        max_results: int = 50
    ) -> List[Tuple[Unit, float]]:
        """
        Find units similar to the given unit using specified metric.

        Available metrics:
        - jaccard_avg, jaccard_min
        - jaccard_no_stop_avg, jaccard_no_stop_min
        - weighted_jaccard_avg, weighted_jaccard_min
        - cosine_tfidf_avg, cosine_tfidf_min
        """
        similar = []

        for other_unit in self.catalog.unit_to_contexts.keys():
            if other_unit == unit:
                continue

            scores = self.compare_unit_pair(unit, other_unit, show_details=False)
            score = scores.get(metric, 0.0)

            if score >= threshold:
                similar.append((other_unit, score))

        # Sort by score descending
        similar.sort(key=lambda x: -x[1])

        return similar[:max_results]


# --- Demo / Test ---

if __name__ == "__main__":
    import argparse
    from catalog import build_catalog_from_cornell

    parser = argparse.ArgumentParser(description="Test similarity metrics")
    parser.add_argument("--max-sentences", type=int, default=10000,
                        help="Maximum sentences to load")
    args = parser.parse_args()

    print("="*70)
    print("SIMILARITY METRICS COMPARISON")
    print("="*70)

    # Build catalog
    catalog = build_catalog_from_cornell(
        max_sentences=args.max_sentences,
        context_size=3,
        max_n=4,
        verbose=True
    )

    print("\n" + "="*70)
    print("INITIALIZING SIMILARITY METRICS")
    print("="*70)

    metrics = SimilarityMetrics(catalog, verbose=True)

    # Test pairs - some should be similar, some should not
    test_pairs = [
        # Should be similar (both phrasal verbs with 'out')
        (("go", "out"), ("get", "out")),
        (("go", "out"), ("hang", "out")),

        # Should NOT be similar (the PDF's false positive example pattern)
        (("go", "out"), ("a", "big")),
        (("go", "out"), ("the", "man")),

        # Should be similar (modal-like)
        (("want", "to"), ("have", "to")),
        (("going", "to"), ("have", "to")),

        # Should be similar (pronouns/determiners in same position)
        (("my", "money"), ("your", "money")),
        (("my", "money"), ("his", "wallet")),

        # Should NOT be similar
        (("my", "money"), ("go", "out")),
        (("i", "love"), ("the", "door")),
    ]

    print("\n" + "="*70)
    print("PAIRWISE COMPARISONS")
    print("="*70)

    for unit1, unit2 in test_pairs:
        # Check both units exist
        if catalog.get_unit_frequency(unit1) == 0:
            print(f"\nSkipping '{' '.join(unit1)}' - not in catalog")
            continue
        if catalog.get_unit_frequency(unit2) == 0:
            print(f"\nSkipping '{' '.join(unit2)}' - not in catalog")
            continue

        metrics.compare_unit_pair(unit1, unit2, show_details=True)

    # Test finding similar units
    print("\n" + "="*70)
    print("FINDING SIMILAR UNITS")
    print("="*70)

    test_units = [("go", "out"), ("want", "to"), ("my", "money")]

    for unit in test_units:
        if catalog.get_unit_frequency(unit) == 0:
            continue

        print(f"\n{'='*60}")
        print(f"Units similar to '{' '.join(unit)}' by different metrics:")
        print(f"{'='*60}")

        for metric in ['jaccard_avg', 'jaccard_no_stop_min', 'cosine_tfidf_min']:
            print(f"\n--- {metric} (threshold=0.15) ---")
            similar = metrics.find_similar_units(unit, metric=metric, threshold=0.15, max_results=15)
            print(f"Found {len(similar)} similar units:")
            for other, score in similar[:15]:
                freq = catalog.get_unit_frequency(other)
                print(f"  {score:.3f}: '{' '.join(other)}' (freq={freq})")
