#!/usr/bin/env python3
"""
Analyze a phrase: parse it and show contextual substitution classes for each node.
Usage:
    python analyze_phrase.py "we should go"
    python analyze_phrase.py "that's a beautiful hat"
"""
import sys
import warnings
warnings.filterwarnings('ignore')
# Import ContextPattern for pickle compatibility
import prototype_topdown_units
from prototype_topdown_units import ContextPattern
from bidir_simple import UnitCatalog, IncrementalBidirParser, print_tree
from collections import defaultdict
import math

def get_all_substitutes_merged(cache: dict, text: str) -> list:
    """
    Get ALL substitutes for a text from ALL cached parses, merged with source tags.

    Returns:
        List of (sub_text, score, source_type, origins) tuples
        source_type: 'aggregation' or 'expansion'
    """
    parses = [(k, v) for k, v in cache.items() if k[0] == text]
    if not parses:
        return []

    merged_subs = {}  # sub_text -> (best_score, source_type, origins)

    for (_, split), cache_value in parses:
        subs = cache_value[0]
        origins = cache_value[4] if len(cache_value) == 5 else None

        # Determine source type from split
        if isinstance(split, tuple) and len(split) == 2 and split[0] == 'expansion':
            source_type = 'expansion'
            parent_split = split[1]  # The parent context split
        elif isinstance(split, tuple) and len(split) == 2:
            # Regular binary split = aggregation from mutual expansion
            source_type = 'aggregation'
            parent_split = split  # The (left, right) split
        elif split is None:
            # Single word, no split
            source_type = 'other'
            parent_split = None
        else:
            source_type = 'other'
            parent_split = split

        # Merge substitutes, keeping best score for each
        for sub_text, score in subs:
            if sub_text not in merged_subs or score > merged_subs[sub_text][0]:
                sub_origins = origins.get(sub_text) if origins else None
                # For expansion, store parent split info; for aggregation, store the split itself
                if not sub_origins and (source_type == 'expansion' or source_type == 'aggregation'):
                    sub_origins = {'split_type': source_type, 'parent_split': parent_split}
                merged_subs[sub_text] = (score, source_type, sub_origins)

    # Convert to list and sort by score
    result = [(text, score, source, origins)
              for text, (score, source, origins) in merged_subs.items()]
    return sorted(result, key=lambda x: -x[1])


def get_best_parse(cache: dict, text: str) -> tuple:
    """
    Get the best parse for a text from cache (lowest energy).

    Returns:
        (substitutes, left_contexts, right_contexts, energy, split, origins) or ([], [], [], inf, None, None) if not found
    """
    # Find all parses for this text
    parses = [(k, v) for k, v in cache.items() if k[0] == text]
    if not parses:
        return ([], [], [], float('inf'), None, None)

    # Return parse with best (lowest) energy
    best_key, best_value = min(parses, key=lambda x: x[1][3])  # x[1][3] is energy

    # Handle variable-length cache entries (some have origins, some don't)
    if len(best_value) == 5:
        # Has origins: (subs, eff_left, eff_right, energy, origins)
        return best_value[:4] + (best_key[1], best_value[4])
    else:
        # No origins: (subs, eff_left, eff_right, energy)
        return best_value + (best_key[1], None)


def run_mutual_expansion(left_text: str, right_text: str,
                         left_eff_left: list, left_eff_right: list,
                         right_eff_left: list, right_eff_right: list,
                         catalog: UnitCatalog,
                         max_class_members: int = 10,
                         scoring: str = 'cosine',
                         verbose: bool = False) -> tuple:
    """
    Run mutual expansion between left and right elements.

    Returns:
        (left_subs, right_subs, left_candidates, right_candidates)
        - left_subs: Initial substitutes for left (from step 1)
        - right_subs: Initial substitutes for right (from step 2)
        - left_candidates: Final LEFT candidates (from step 4)
        - right_candidates: Final RIGHT candidates (from step 3)
    """
    # Skip if either side has no context fillers
    if not left_eff_right and not right_eff_left:
        return ([], [], [], [])

    # Step 1: Get substitutes for LEFT
    if verbose:
        print(f"      [STEP 1] Get substitutes for LEFT: \"{left_text}\"")
        print(f"        Candidates: {len(right_eff_left)} words from RIGHT's left_words")

    all_left_subs = catalog.gpu_contextual_candidates(
        target=left_text,
        candidates=right_eff_left,
        max_results=max_class_members * 3,
        scoring=scoring,
        trace=False
    )
    all_left_subs = [(t, s) for t, s in all_left_subs if t != left_text]
    left_subs = all_left_subs[:max_class_members]

    # Collect right_words from left substitutes → candidates for RIGHT
    context_word_counts = {}
    n_subs_with_pattern = 0
    for sub_text, _ in left_subs:
        sub_pattern = catalog.get_unit(sub_text)
        if sub_pattern:
            n_subs_with_pattern += 1
            for word in sub_pattern.right_words.keys():
                context_word_counts[word] = context_word_counts.get(word, 0) + 1

    min_share = max(2, int(n_subs_with_pattern * 0.3)) if n_subs_with_pattern > 0 else 2
    right_contexts_of_left_element = {w for w, c in context_word_counts.items() if c >= min_share}

    # Step 2: Get substitutes for RIGHT
    if verbose:
        print(f"      [STEP 2] Get substitutes for RIGHT: \"{right_text}\"")
        print(f"        Candidates: {len(left_eff_right)} words from LEFT's right_words")

    all_right_subs = catalog.gpu_contextual_candidates(
        target=right_text,
        candidates=left_eff_right,
        max_results=max_class_members * 3,
        scoring=scoring,
        trace=False
    )
    all_right_subs = [(t, s) for t, s in all_right_subs if t != right_text]
    right_subs = all_right_subs[:max_class_members]

    # Collect left_words from right substitutes → candidates for LEFT
    context_word_counts_r = {}
    n_rsubs_with_pattern = 0
    for sub_text, _ in right_subs:
        sub_pattern = catalog.get_unit(sub_text)
        if sub_pattern:
            n_rsubs_with_pattern += 1
            for word in sub_pattern.left_words.keys():
                context_word_counts_r[word] = context_word_counts_r.get(word, 0) + 1

    min_share_r = max(2, int(n_rsubs_with_pattern * 0.3)) if n_rsubs_with_pattern > 0 else 2
    left_contexts_of_right_element = {w for w, c in context_word_counts_r.items() if c >= min_share_r}

    # Step 3: Score RIGHT candidates
    right_candidates = catalog.gpu_contextual_candidates(
        target=right_text,
        candidates=list(right_contexts_of_left_element),
        max_results=max_class_members + 1,
        scoring=scoring,
        trace=False
    )
    right_candidates = [(t, s) for t, s in right_candidates if t != right_text][:max_class_members]

    # Step 4: Score LEFT candidates
    left_candidates = catalog.gpu_contextual_candidates(
        target=left_text,
        candidates=list(left_contexts_of_right_element),
        max_results=max_class_members + 1,
        scoring=scoring,
        trace=False
    )
    left_candidates = [(t, s) for t, s in left_candidates if t != left_text][:max_class_members]

    return (left_subs, right_subs, left_candidates, right_candidates)


def aggregate_contexts_from_constituents(text: str, subparse_cache: dict, catalog: UnitCatalog, verbose: bool = False) -> tuple:
    """
    For multi-word units, aggregate contexts from COMBINATIONS of substitutes across binary splits.

    For "i want to", we test all binary parses:
    - ("i" "want to"): aggregate from combinations of i-subs × want-to-subs
    - ("i want" "to"): aggregate from combinations of i-want-subs × to-subs

    For each (left_sub, right_sub) pair:
    - Collect right_words from left_sub (what can follow left_sub)
    - Collect left_words from right_sub (what can precede right_sub)

    This captures actual alternative phrasings, not just constituent contexts.
    """
    tokens = text.split()
    if len(tokens) == 1:
        # Single word - use its own pattern
        pattern = catalog.get_unit(text)
        eff_left = [w for w, c in pattern.left_words.most_common(50) if c >= 3] if pattern else []
        eff_right = [w for w, c in pattern.right_words.most_common(50) if c >= 3] if pattern else []
        return (eff_left, eff_right)

    # Multi-word: aggregate from combinations across all binary parses
    all_left = set()
    all_right = set()

    # Include the unit's own context if it exists
    pattern = catalog.get_unit(text)
    own_left_count = 0
    own_right_count = 0
    if pattern:
        own_left = [w for w, c in pattern.left_words.most_common(50) if c >= 3]
        own_right = [w for w, c in pattern.right_words.most_common(50) if c >= 3]
        all_left.update(own_left)
        all_right.update(own_right)
        own_left_count = len(own_left)
        own_right_count = len(own_right)

    # Test all binary splits of this multi-word unit
    combination_count = 0
    for split in range(1, len(tokens)):
        left_text = " ".join(tokens[:split])
        right_text = " ".join(tokens[split:])

        # Get cached substitution classes for both sides (best parse)
        left_subs, left_eff_left, left_eff_right, _, _, _ = get_best_parse(subparse_cache, left_text)
        right_subs, right_eff_left, right_eff_right, _, _, _ = get_best_parse(subparse_cache, right_text)

        # Bootstrap constituents if not cached
        if not left_subs:
            if len(left_text.split()) == 1:
                # Single word: use identity substitute
                left_pattern = catalog.get_unit(left_text)
                if left_pattern:
                    left_subs = [(left_text, 1.0)]
                    left_eff_left = [w for w, c in left_pattern.left_words.most_common(50) if c >= 3]
                    left_eff_right = [w for w, c in left_pattern.right_words.most_common(50) if c >= 3]
            else:
                # Multi-word: recursively aggregate contexts and run mutual expansion
                left_eff_left, left_eff_right = aggregate_contexts_from_constituents(
                    left_text, subparse_cache, catalog, verbose=False
                )
                # Generate substitutes through mutual expansion of its constituents
                # For now, use identity as placeholder - will be generated in next iteration
                left_subs = [(left_text, 1.0)]

        if not right_subs:
            if len(right_text.split()) == 1:
                # Single word: use identity substitute
                right_pattern = catalog.get_unit(right_text)
                if right_pattern:
                    right_subs = [(right_text, 1.0)]
                    right_eff_left = [w for w, c in right_pattern.left_words.most_common(50) if c >= 3]
                    right_eff_right = [w for w, c in right_pattern.right_words.most_common(50) if c >= 3]
            else:
                # Multi-word: recursively aggregate contexts and run mutual expansion
                right_eff_left, right_eff_right = aggregate_contexts_from_constituents(
                    right_text, subparse_cache, catalog, verbose=False
                )
                # Generate substitutes through mutual expansion of its constituents
                # For now, use identity as placeholder - will be generated in next iteration
                right_subs = [(right_text, 1.0)]

        # Now run mutual expansion between left and right to generate actual substitutes
        if left_eff_left or left_eff_right or right_eff_left or right_eff_right:
            if verbose:
                print(f"      [AGGREGATION] Running mutual expansion for \"{text}\" split: (\"{left_text}\", \"{right_text}\")")
            left_subs_exp, right_subs_exp, _, _ = run_mutual_expansion(
                left_text, right_text,
                left_eff_left, left_eff_right,
                right_eff_left, right_eff_right,
                catalog,
                max_class_members=10,
                scoring='cosine',
                verbose=verbose
            )
            # Use generated substitutes if we got any, otherwise keep identity
            if left_subs_exp:
                left_subs = left_subs_exp
            if right_subs_exp:
                right_subs = right_subs_exp

        # Skip if either side still has no substitutes
        if not left_subs or not right_subs:
            continue

        # Aggregate contexts from all combinations of left_subs × right_subs
        # Also collect these combinations as substitutes to cache
        all_combinations = []
        combination_origins = {}  # Track where each combination came from

        for left_sub_text, left_score in left_subs:
            for right_sub_text, right_score in right_subs:
                # Form the joined phrase from this combination
                combined_text = f"{left_sub_text} {right_sub_text}"

                # Look up this complete alternative phrasing in the catalog
                combined_pattern = catalog.get_unit(combined_text)
                if not combined_pattern:
                    continue

                # Collect contexts from this alternative complete phrasing
                # left_words: what can precede this entire phrase
                for word in combined_pattern.left_words.keys():
                    all_left.add(word)

                # right_words: what can follow this entire phrase
                for word in combined_pattern.right_words.keys():
                    all_right.add(word)

                combination_count += 1

                # NEW: Also save this combination as a potential substitute
                combo_score = left_score * right_score
                all_combinations.append((combined_text, combo_score))

                # NEW: Track origin for debugging/tracing
                # Show full substitution path: what → substitute
                combination_origins[combined_text] = {
                    'left_source': left_text,      # e.g., "hat"
                    'left_sub': left_sub_text,     # e.g., "together"
                    'left_score': left_score,
                    'right_source': right_text,    # e.g., "again"
                    'right_sub': right_sub_text,   # e.g., "and"
                    'right_score': right_score
                }

        # NEW: Cache the combinations as substitutes for this text
        # This makes them available when this span is used in larger contexts
        if all_combinations and len(tokens) > 1:
            # Sort by score and cache
            all_combinations = sorted(all_combinations, key=lambda x: -x[1])[:30]
            # Use a special marker to indicate these came from aggregation
            agg_energy = -len(all_combinations) * 0.1  # Simple energy estimate

            # Store with origin information for tracing
            subparse_cache[(text, ('aggregation', tuple(tokens)))] = (
                all_combinations, list(all_left), list(all_right), agg_energy, combination_origins
            )
            if verbose:
                print(f"      Cached {len(all_combinations)} combinations as substitutes for \"{text}\"")

    if verbose and len(tokens) > 1:
        print(f"      Context aggregation for \"{text}\": "
              f"own=({own_left_count}L, {own_right_count}R), "
              f"{combination_count} combinations → "
              f"enriched=({len(all_left)}L, {len(all_right)}R)")

    return (list(all_left), list(all_right))

def compute_energy(num_substitutes: int, consensus_score: float) -> float:
    """
    Compute energy from substitution class properties.

    Lower energy = better defined meaning:
    - More substitutes → lower energy (well-supported pattern)
    - Higher consensus → lower energy (converged meaning)

    Args:
        num_substitutes: Number of substitutes found
        consensus_score: Average context overlap consensus (0-1)

    Returns:
        Energy value (lower is better)
    """
    # Avoid log(0)
    num_subs = max(1, num_substitutes)
    consensus = max(0.01, consensus_score)

    # Energy combines both factors (equal weighting)
    # Negative because more subs/consensus = lower energy
    energy = -math.log(num_subs) - math.log(consensus)

    return energy

def compute_consensus_score(substitutes: list, catalog: UnitCatalog, target_text: str) -> float:
    """
    Compute consensus score: how much substitutes agree on their contexts.

    Returns:
        Consensus score 0-1 (1 = perfect agreement)
    """
    if not substitutes:
        return 0.01

    target_pattern = catalog.get_unit(target_text)
    if not target_pattern:
        return 0.01

    # Get target's context words
    target_left = set(target_pattern.left_words.keys())
    target_right = set(target_pattern.right_words.keys())
    all_target_context = target_left | target_right

    if not all_target_context:
        return 0.01

    # For each context word, count how many substitutes share it
    word_counts = {}
    for sub_text, _ in substitutes:
        sub_pattern = catalog.get_unit(sub_text)
        if sub_pattern:
            sub_contexts = set(sub_pattern.left_words.keys()) | set(sub_pattern.right_words.keys())
            for word in sub_contexts & all_target_context:
                word_counts[word] = word_counts.get(word, 0) + 1

    if not word_counts:
        return 0.01

    # Average frequency (normalized by num substitutes)
    avg_frequency = sum(word_counts.values()) / (len(word_counts) * len(substitutes))

    return min(1.0, avg_frequency)

def analyze(phrase: str, catalog: UnitCatalog, parser: IncrementalBidirParser,
            max_class_members: int = 500, scoring: str = 'cosine', external_left: list = [], external_right: list = []):
    """Analyze a phrase and display results."""
    tokens = phrase.lower().split()
    print("=" * 70)
    print(f'INPUT: "{phrase}"')
    print(f'SCORING: {scoring}')
    print("=" * 70)

    # Incremental processing
    print("\n[1] INCREMENTAL BOTTOM-UP ANALYSIS (SEQUENTIAL PRESENTATION)")
    print("-" * 40)
    parser.reset()  # Clear for new sequence
    subparse_cache = {}  # prefix_text -> (subs, eff_left, eff_right)
    html_buffer = []  # NEW: Buffer HTML during parsing for efficiency
    for k, word in enumerate(tokens, 1):
        parser.add_word(word)  # Add word sequentially
        print(f"\nAdding word '{word}' at position {k}")

        # Parse ALL subspans ending at position k (from shortest to longest)
        # This ensures right-side constituents like "found my" are available for splits
        for start_pos in range(k):  # 0 to k-1
            span_tokens = tokens[start_pos:k]
            span_text = " ".join(span_tokens)
            span_length = len(span_tokens)

            print(f"  Parsing span [{start_pos}:{k}]: \"{span_text}\"")

            # Bootstrap single-word spans
            if span_length == 1:
                # For single word, no candidates available yet - just cache the unit itself
                subs = []
                # Use aggregation (handles both single and multi-word gracefully)
                eff_left, eff_right = aggregate_contexts_from_constituents(
                    span_text, subparse_cache, catalog
                )
                # No split for single word, energy is high (no substitutes)
                energy = float('inf')
                subparse_cache[(span_text, None)] = (subs, eff_left, eff_right, energy)

                # Add to GPU index for dynamic scoring
                from collections import Counter
                left_context_dict = Counter({w: 1 for w in eff_left})
                right_context_dict = Counter({w: 1 for w in eff_right})
                catalog.add_unit_to_gpu_index(span_text, left_context_dict, right_context_dict)

                print(f"    Bootstrap subs for \"{span_text}\": {subs}")
                continue

            # Test all pairwise splits (binary mappings), building right-to-left
            # Process in REVERSE order so constituents are cached before they're needed
            for split_pos in range(start_pos + span_length - 1, start_pos, -1):  # right-to-left within span
                left_text = " ".join(tokens[start_pos:split_pos])
                right_text = " ".join(tokens[split_pos:k])
                print(f"    Testing split: (\"{left_text}\" \"{right_text}\")")

                # Get cached parses for left and right (best energy)
                left_subs, left_eff_left, left_eff_right, left_energy, left_split, _ = get_best_parse(subparse_cache, left_text)
                right_subs, right_eff_left, right_eff_right, right_energy, right_split, _ = get_best_parse(subparse_cache, right_text)

                # Bootstrap right if not cached yet
                if not right_subs:
                    # Use constituent aggregation for multi-word units
                    right_eff_left, right_eff_right = aggregate_contexts_from_constituents(
                        right_text, subparse_cache, catalog, verbose=True
                    )
                    # For initial bootstrap, right_subs is just the unit itself
                    right_subs = [(right_text, 1.0)]
                    # Bootstrap has no split (identity), energy is high
                    right_energy = float('inf')
                    subparse_cache[(right_text, None)] = (right_subs, right_eff_left, right_eff_right, right_energy)

                    # Add to GPU index for dynamic scoring
                    from collections import Counter
                    left_context_dict = Counter({w: 1 for w in right_eff_left})
                    right_context_dict = Counter({w: 1 for w in right_eff_right})
                    catalog.add_unit_to_gpu_index(right_text, left_context_dict, right_context_dict)

                # MUTUAL EXPANSION: Only proceed if we have context from BOTH sides
                # This ensures we have actual fillers to seed the expansion

                print(f"      Left context available: {len(left_eff_left)} left, {len(left_eff_right)} right")
                print(f"      Right context available: {len(right_eff_left)} left, {len(right_eff_right)} right")

                # Skip mutual expansion if either side has no context fillers
                if not left_eff_right and not right_eff_left:
                    print(f"      Skipping mutual expansion: no context fillers from either side yet")
                    # Cache the span with just the units themselves
                    combined_subs = [(left_text, 1.0), (right_text, 1.0)]
                    combined_energy = float('inf')  # No real substitutes
                    subparse_cache[(span_text, (left_text, right_text))] = (combined_subs, left_eff_left, left_eff_right, combined_energy)
                    html_buffer.append((span_text, (left_text, right_text), combined_energy, span_length))

                    # Add to GPU index for dynamic scoring
                    from collections import Counter
                    left_context_dict = Counter({w: 1 for w in left_eff_left})
                    right_context_dict = Counter({w: 1 for w in left_eff_right})
                    catalog.add_unit_to_gpu_index(span_text, left_context_dict, right_context_dict)

                    continue

                # Step 1: Get substitutes for LEFT
                # Candidates ARE right_eff_left (left_words of RIGHT = words in LEFT's position)
                # Score them by similarity to LEFT's context patterns
                print(f"      [STEP 1] Get substitutes for LEFT: \"{left_text}\"")
                print(f"        Candidates: {len(right_eff_left)} words from RIGHT's left_words")
                all_left_subs = catalog.gpu_contextual_candidates(
                    target=left_text,
                    candidates=right_eff_left,
                    max_results=max_class_members * 3,
                    scoring=scoring,
                    trace=True  # Debug scoring
                )
                all_left_subs = [(t, s) for t, s in all_left_subs if t != left_text]
                left_subs_for_expansion = all_left_subs[:max_class_members]
                print(f"        Found {len(all_left_subs)} scored, keeping top {len(left_subs_for_expansion)}")

                # Collect right_words from left substitutes → candidates for RIGHT
                # Context consensus: only keep words shared by multiple substitutes
                print(f"      Left substitutes: {[t for t, _ in left_subs_for_expansion]}")
                context_word_counts = {}  # word → how many substitutes have it as right_word
                n_subs_with_pattern = 0
                for sub_text, _ in left_subs_for_expansion:
                    sub_pattern = catalog.get_unit(sub_text)
                    if sub_pattern:
                        n_subs_with_pattern += 1
                        for word in sub_pattern.right_words.keys():
                            context_word_counts[word] = context_word_counts.get(word, 0) + 1

                # Keep words shared by ≥ 30% of substitutes (minimum 2)
                min_share = max(2, int(n_subs_with_pattern * 0.3))
                right_contexts_of_left_element = {w for w, c in context_word_counts.items() if c >= min_share}
                print(f"      Context consensus: {len(context_word_counts)} raw → "
                      f"{len(right_contexts_of_left_element)} shared (≥{min_share}/{n_subs_with_pattern} subs)")

                # Step 2: Get substitutes for RIGHT
                # Candidates ARE left_eff_right (right_words of LEFT = words in RIGHT's position)
                # Score them by similarity to RIGHT's context patterns
                print(f"    [STEP 2] Get substitutes for RIGHT: \"{right_text}\"")
                print(f"      Candidates: {len(left_eff_right)} words from LEFT's right_words")
                all_right_subs = catalog.gpu_contextual_candidates(
                    target=right_text,
                    candidates=left_eff_right,
                    max_results=max_class_members * 3,
                    scoring=scoring
                )
                all_right_subs = [(t, s) for t, s in all_right_subs if t != right_text]
                right_subs_for_expansion = all_right_subs[:max_class_members]
                print(f"      Found {len(all_right_subs)} scored, keeping top {len(right_subs_for_expansion)}")

                # Collect left_words from right substitutes → candidates for LEFT
                # Context consensus: only keep words shared by multiple substitutes
                print(f"      Right substitutes: {[t for t, _ in right_subs_for_expansion]}")
                context_word_counts_r = {}  # word → how many substitutes have it as left_word
                n_rsubs_with_pattern = 0
                for sub_text, _ in right_subs_for_expansion:
                    sub_pattern = catalog.get_unit(sub_text)
                    if sub_pattern:
                        n_rsubs_with_pattern += 1
                        for word in sub_pattern.left_words.keys():
                            context_word_counts_r[word] = context_word_counts_r.get(word, 0) + 1

                # Keep words shared by ≥ 30% of substitutes (minimum 2)
                min_share_r = max(2, int(n_rsubs_with_pattern * 0.3))
                left_contexts_of_right_element = {w for w, c in context_word_counts_r.items() if c >= min_share_r}
                print(f"      Context consensus: {len(context_word_counts_r)} raw → "
                      f"{len(left_contexts_of_right_element)} shared (≥{min_share_r}/{n_rsubs_with_pattern} subs)")

                # Step 3: Score RIGHT candidates (consensus-filtered right_words from left subs)
                # These are words that follow left substitutes, filtered to shared ones
                # Score by similarity to RIGHT's context patterns
                print(f"    [STEP 3] Score RIGHT candidates by matching RIGHT's constraints")
                print(f"      Candidates: {len(right_contexts_of_left_element)} words (right_contexts of left element)")
                print(f"      Sample: {sorted(list(right_contexts_of_left_element))[:10]}")
                right_candidates = catalog.gpu_contextual_candidates(
                    target=right_text,
                    candidates=list(right_contexts_of_left_element),
                    max_results=max_class_members + 1,
                    scoring=scoring
                )
                right_candidates = [(t, s) for t, s in right_candidates if t != right_text][:max_class_members]
                print(f"      Found {len(right_candidates)} scored candidates for RIGHT")

                # Step 4: Score LEFT candidates (consensus-filtered left_words from right subs)
                # These are words that precede right substitutes, filtered to shared ones
                # Score by similarity to LEFT's context patterns
                print(f"    [STEP 4] Score LEFT candidates by matching LEFT's constraints")
                print(f"      Candidates: {len(left_contexts_of_right_element)} words (left_contexts of right element)")
                print(f"      Sample: {sorted(list(left_contexts_of_right_element))[:10]}")
                left_candidates = catalog.gpu_contextual_candidates(
                    target=left_text,
                    candidates=list(left_contexts_of_right_element),
                    max_results=max_class_members + 1,
                    scoring=scoring
                )
                left_candidates = [(t, s) for t, s in left_candidates if t != left_text][:max_class_members]
                print(f"      Found {len(left_candidates)} scored candidates for LEFT")

                print(f"    Left candidates: {left_candidates[:5]}")
                print(f"    Right candidates: {right_candidates[:5]}")

                # NEW: Enhance right_candidates with multi-word combinations filtered on external context
                # If right_text is multi-word and we have an external left context (left_text),
                # add combinations of right's constituents that can follow left_text
                if ' ' in right_text and ' ' not in left_text:  # right is multi-word, left is single-word
                    print(f"    [ENHANCEMENT] Adding multi-word combinations for RIGHT \"{right_text}\" filtered on external context \"{left_text}\"")

                    # Find constituent splits for right_text in cache
                    right_constituent_splits = [(k, v) for k, v in subparse_cache.items()
                                               if k[0] == right_text and isinstance(k[1], tuple) and len(k[1]) == 2]

                    if right_constituent_splits:
                        # Use the first split (could try all splits and merge)
                        split_key, split_value = right_constituent_splits[0]
                        right_split_left, right_split_right = split_key[1]

                        print(f"      Found constituent split: (\"{right_split_left}\", \"{right_split_right}\")")

                        # Get substitutes for each constituent
                        r_left_subs, _, _, _, _, _ = get_best_parse(subparse_cache, right_split_left)
                        r_right_subs, _, _, _, _, _ = get_best_parse(subparse_cache, right_split_right)

                        if r_left_subs and r_right_subs:
                            print(f"      Forming combinations from {len(r_left_subs)} × {len(r_right_subs)} constituent substitutes")
                            multi_word_candidates = []

                            for r_left_text, r_left_score in r_left_subs[:10]:
                                for r_right_text, r_right_score in r_right_subs[:10]:
                                    combo_text = f"{r_left_text} {r_right_text}"

                                    # Check if combination exists in catalog
                                    combo_pattern = catalog.get_unit(combo_text)
                                    if not combo_pattern:
                                        continue

                                    # Filter on external context: can left_text precede this combination?
                                    if left_text in combo_pattern.left_words:
                                        combo_score = r_left_score * r_right_score
                                        multi_word_candidates.append((combo_text, combo_score))

                            print(f"      Found {len(multi_word_candidates)} multi-word combinations compatible with \"{left_text}\"")
                            if multi_word_candidates:
                                # Merge with right_candidates
                                right_candidates.extend(multi_word_candidates)
                                right_candidates = sorted(right_candidates, key=lambda x: -x[1])[:max_class_members * 2]
                                print(f"      Enhanced RIGHT candidates (now {len(right_candidates)} total)")
                                print(f"      Sample multi-word: {multi_word_candidates[:3]}")

                # Compute energies for left and right based on their substitution classes
                left_consensus = compute_consensus_score(left_subs_for_expansion, catalog, left_text)
                left_energy = compute_energy(len(left_subs_for_expansion), left_consensus)

                right_consensus = compute_consensus_score(right_subs_for_expansion, catalog, right_text)
                right_energy = compute_energy(len(right_subs_for_expansion), right_consensus)

                print(f"    LEFT energy: {left_energy:.2f} (n={len(left_subs_for_expansion)}, consensus={left_consensus:.3f})")
                print(f"    RIGHT energy: {right_energy:.2f} (n={len(right_subs_for_expansion)}, consensus={right_consensus:.3f})")

                # Update cache for whole prefix
                # Combine left and right candidates as substitutes for the full span
                # Form cartesian product: each left substitute paired with each right substitute
                combined_candidates = []
                external_filtered_candidates = []
                origins_dict = {}  # Track origin of each substitute

                # DEBUG counters
                total_combinations_tried = 0
                combinations_in_catalog = 0
                combinations_with_external_context = 0

                for left_sub, left_score in left_candidates:
                    for right_sub, right_score in right_candidates:
                        # Form the combined phrase
                        combined_text = f"{left_sub} {right_sub}"
                        total_combinations_tried += 1

                        # Only keep combinations that exist in catalog
                        # This avoids synthetic phrases with no observed contexts
                        combined_pattern = catalog.get_unit(combined_text)
                        if not combined_pattern:
                            continue

                        combinations_in_catalog += 1

                        # Combine scores (multiply to get joint probability-like score)
                        combined_score = left_score * right_score
                        combined_candidates.append((combined_text, combined_score))

                        # Track origin: which left and right subs were combined
                        origins_dict[combined_text] = {
                            'left_sub': left_sub,
                            'left_score': left_score,
                            'left_source': right_text,  # Left subs come from left context of right element
                            'right_sub': right_sub,
                            'right_score': right_score,
                            'right_source': left_text  # Right subs come from right context of left element
                        }

                        # NEW: Also filter multi-word combinations on external context
                        # Check if any substitute of the left element can precede this combination
                        # This tests compatibility with the substitution class, not just the head phrase
                        compatible_left_subs = [lsub for lsub, _ in left_candidates
                                               if lsub in combined_pattern.left_words]
                        if compatible_left_subs:
                            # Track which external context this came from
                            combinations_with_external_context += 1
                            context_info = f"filtered_on_L_subs:{','.join(compatible_left_subs[:3])}"
                            external_filtered_candidates.append((combined_text, combined_score, context_info))

                print(f"    Combination statistics: {total_combinations_tried} tried, {combinations_in_catalog} in catalog, {combinations_with_external_context} with external context")
                if combinations_in_catalog > 0 and combinations_in_catalog <= 10:
                    print(f"    Combinations found in catalog: {[t for t, s in combined_candidates[:10]]}")

                # Sort by score and keep top (increase limit to handle sparsity)
                combined_candidates = sorted(combined_candidates, key=lambda x: -x[1])[:max_class_members * 3]
                external_filtered_candidates = sorted(external_filtered_candidates, key=lambda x: -x[1])[:max_class_members * 2]

                print(f"    Found {len(combined_candidates)} combined candidates, {len(external_filtered_candidates)} externally filtered")

                # DEBUG: Show what types of candidates we have
                single_word_combos = [t for t, s in combined_candidates if ' ' not in t]
                multi_word_combos = [t for t, s in combined_candidates if ' ' in t]
                print(f"    Combined candidates breakdown: {len(single_word_combos)} single-word, {len(multi_word_combos)} multi-word")
                if multi_word_combos:
                    print(f"    Sample multi-word combinations: {multi_word_combos[:5]}")

                # Merge current substitutes with externally filtered multi-word combinations
                # Convert to uniform format first: (text, score, context_info)
                all_substitutes_with_context = []

                # Add combinations from mutual expansion
                for text, score in combined_candidates:
                    all_substitutes_with_context.append((text, score, f"from_mutual_expansion"))

                # Add externally filtered multi-word units (may overlap with above)
                for text, score, context_info in external_filtered_candidates:
                    # Check if not already added
                    if not any(t == text for t, _, _ in all_substitutes_with_context):
                        all_substitutes_with_context.append((text, score, context_info))

                # Sort by score and keep top
                all_substitutes_with_context = sorted(all_substitutes_with_context, key=lambda x: -x[1])[:max_class_members * 3]

                # Convert back to (text, score) format for energy computation
                combined_candidates = [(text, score) for text, score, _ in all_substitutes_with_context]

                # Compute energy for the combined span
                combined_consensus = compute_consensus_score(combined_candidates, catalog, span_text)
                combined_energy = compute_energy(len(combined_candidates), combined_consensus)

                # Update context words from both sides
                eff_left = list(set(left_eff_left) | set(right_eff_left))
                eff_right = list(set(left_eff_right) | set(right_eff_right))

                # Cache with split structure, energy, and origins
                subparse_cache[(span_text, (left_text, right_text))] = (combined_candidates, eff_left, eff_right, combined_energy, origins_dict)

                # NEW: Collect parse info for HTML generation
                html_buffer.append((span_text, (left_text, right_text), combined_energy, span_length))

                # Also cache the constituent substitutes from this split
                # Use a marker ('from_expansion', parent_split) to indicate these came from mutual expansion
                # This allows reuse when processing longer phrases
                if left_subs_for_expansion:
                    # Cache substitutes generated for left constituent during this expansion
                    subparse_cache[(left_text, ('expansion', (left_text, right_text)))] = (
                        left_subs_for_expansion, left_eff_left, left_eff_right, left_energy
                    )

                if right_subs_for_expansion:
                    # Cache substitutes generated for right constituent during this expansion
                    subparse_cache[(right_text, ('expansion', (left_text, right_text)))] = (
                        right_subs_for_expansion, right_eff_left, right_eff_right, right_energy
                    )

                print(f"      COMBINED energy: {combined_energy:.2f} (n={len(combined_candidates)}, consensus={combined_consensus:.3f})")

                # Show top substitutes with context information
                print(f"      Top substitutes with context info:")
                for i, (text, score, context_info) in enumerate(all_substitutes_with_context[:5], 1):
                    print(f"        {i}. \"{text}\" (score={score:.3f}) [{context_info}]")

                # Add to GPU index for dynamic scoring
                # Convert context lists to dicts with equal weights
                from collections import Counter
                left_context_dict = Counter({w: 1 for w in eff_left})
                right_context_dict = Counter({w: 1 for w in eff_right})
                catalog.add_unit_to_gpu_index(span_text, left_context_dict, right_context_dict)

                print(f"      Cache updated for \"{span_text}\" split (\"{left_text}\", \"{right_text}\"): {len(combined_candidates)} combined candidates")

                # Store energies for this split in the parser
                # The parser will use these to compute span energies
                parser.store_split_energy(left_text, right_text, left_energy + right_energy)

    tree = parser.get_current_parse()

    # Final sections (original)
    print("\n[2] PARSE TREE")
    print("-" * 40)

    # Display the actual parse tree structure with energies
    if tree:
        def print_parse_tree(node, prefix="", is_left=None):
            """Recursively print parse tree with energies and frequencies."""
            # Determine connector based on position
            if is_left is None:
                connector = "└─ "
                extension = "   "
            elif is_left:
                connector = "├─ "
                extension = "│  "
            else:
                connector = "└─ "
                extension = "   "

            # Check if we should use the stored split instead of the unit
            # This shows the internal structure even when a unit was chosen
            use_split = (node.left is None and node.right is None and
                        node.best_split_left is not None and node.best_split_right is not None)

            if use_split:
                # Show as split using the alternative split structure
                print(f"{prefix}{connector}[SPLIT] E={node.best_split_energy:.2f} \"{node.span.text}\"")
                print_parse_tree(node.best_split_left, prefix + extension, is_left=True)
                print_parse_tree(node.best_split_right, prefix + extension, is_left=False)
            elif node.left is None and node.right is None:
                # True leaf node - this is a unit with no possible splits
                pattern = catalog.get_unit(node.span.text)
                freq = pattern.count if pattern else 0
                print(f"{prefix}{connector}[UNIT] E={node.energy:.2f} freq={freq} \"{node.span.text}\"")
            else:
                # Internal node - this is a split that was actually chosen
                print(f"{prefix}{connector}[SPLIT] E={node.energy:.2f} \"{node.span.text}\"")
                if node.left and node.right:
                    # Both children exist
                    print_parse_tree(node.left, prefix + extension, is_left=True)
                    print_parse_tree(node.right, prefix + extension, is_left=False)
                elif node.left:
                    # Only left child
                    print_parse_tree(node.left, prefix + extension, is_left=False)
                elif node.right:
                    # Only right child
                    print_parse_tree(node.right, prefix + extension, is_left=False)

        print_parse_tree(tree)
    else:
        print("No parse tree generated")

    print("\n[2b] ALL PARSE CANDIDATES EVALUATED (from mutual expansion)")
    print("-" * 40)

    # Show all spans that were parsed, with their candidates and frequencies
    print("CANDIDATE SUBSTITUTES BY SPAN:")
    # Group by text and show best parse for each
    texts_seen = set()
    for (span_text, split), cache_value in sorted(subparse_cache.items(), key=lambda x: len(x[0][0].split())):
        if span_text in texts_seen:
            continue
        texts_seen.add(span_text)

        # Handle variable-length cache entries (some have origins, some don't)
        if len(cache_value) == 5:
            subs, eff_left, eff_right, energy, origins = cache_value
        else:
            subs, eff_left, eff_right, energy = cache_value

        # Get best parse for this text
        best_subs, _, _, best_energy, best_split, _ = get_best_parse(subparse_cache, span_text)

        pattern = catalog.get_unit(span_text)
        freq = pattern.count if pattern else 0

        span_tokens = span_text.split()
        indent = "  " * (len(span_tokens) - 1)

        print(f"{indent}[{len(span_tokens)}-GRAM] \"{span_text}\" freq={freq} energy={best_energy:.2f} split={best_split}")

        # Show top 3 candidates for this span
        if best_subs:
            for sub_text, score in best_subs[:3]:
                sub_freq = catalog.get_unit(sub_text)
                sub_freq = sub_freq.count if sub_freq else 0
                print(f"{indent}  → \"{sub_text}\" (score={score:.6f}, freq={sub_freq})")
        else:
            print(f"{indent}  (Substitutes dependent on external context, so not cached)")

    print("\n[3] CONTEXTUAL SUBSTITUTION CLASSES")
    print("-" * 40)

    # Display substitution classes for nodes in the final parse tree
    # Use cached candidates from mutual expansion during incremental analysis
    if tree:
        def show_node_subs(node, depth=0):
            """Recursively show substitution classes for each node in parse tree."""
            indent = "  " * depth
            span_text = node.span.text

            # Look up this span in the cache - should have candidates from mutual expansion
            subs, eff_left, eff_right, energy, split, _ = get_best_parse(subparse_cache, span_text)

            print(f"{indent}Span: \"{span_text}\"")
            if subs:
                print(f"{indent}  Substitution class (from mutual expansion):")
                for sub_text, score in subs[:5]:  # Show top 5
                    print(f"{indent}    - {str(sub_text):<25s} (score: {score:.6f})")
            else:
                print(f"{indent}  (No substitutes found in cache)")

            # Recursively show left and right children
            if node.left:
                show_node_subs(node.left, depth + 1)
            if node.right:
                show_node_subs(node.right, depth + 1)

        show_node_subs(tree)

        # Also show all competing split candidates that were evaluated
        print(f"\n  All mutual expansion candidates evaluated:")
        texts_seen = set()
        for (span_text, split) in sorted(subparse_cache.keys(), key=lambda x: x[0]):
            if span_text in texts_seen:
                continue
            texts_seen.add(span_text)
            subs, _, _, energy, _, _ = get_best_parse(subparse_cache, span_text)
            if subs:
                print(f"    {span_text:<25s}: {len(subs)} candidates, top score: {subs[0][1]:.6f}, energy: {energy:.2f}")
    else:
        print("No parse tree to display")

    # Return tree, cache, and parse info buffer for HTML export
    print(f"\nCollected {len(html_buffer)} parses for HTML generation")
    return tree, subparse_cache, html_buffer

def export_html_tree(phrase: str, tree, subparse_cache: dict, catalog: UnitCatalog, parse_buffer: list, output_file: str = "parse_tree.html"):
    """
    Export parse tree as interactive HTML file.

    Args:
        phrase: Input phrase
        tree: Parse tree
        subparse_cache: Cache with all parses
        catalog: Unit catalog
        output_file: Output HTML filename
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Parse Tree: {phrase}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .tree {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .node {{ margin: 10px 0; padding: 10px; border-left: 3px solid #4CAF50; background: #fafafa; }}
        .node:hover {{ background: #f0f0f0; cursor: pointer; }}
        .span-text {{ font-weight: bold; color: #2196F3; cursor: pointer; }}
        .span-text:hover {{ text-decoration: underline; background: #e3f2fd; }}
        .energy {{ color: #FF5722; font-size: 0.9em; }}
        .split-info {{ color: #666; font-size: 0.85em; font-style: italic; }}
        .substitutes {{ display: none; margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 4px; }}
        .substitutes.visible {{ display: block; }}
        .sub-item {{ margin: 5px 0; padding: 5px; background: white; border-radius: 3px; }}
        .score {{ color: #666; font-size: 0.9em; }}
        .substitute-text {{ font-weight: bold; position: relative; cursor: help; }}
        .substitute-text:hover {{ background: #ffeb3b; }}
        .sub-aggregation {{ color: #000; }}
        .sub-expansion {{ color: #f57c00; }}
        .sub-other {{ color: #666; }}
        .left-context {{ color: #d32f2f; font-size: 0.85em; }}
        .right-context {{ color: #388e3c; font-size: 0.85em; }}
        .context-label {{ font-weight: bold; font-size: 0.8em; color: #666; }}
        .origin-tooltip {{ display: none; position: absolute; bottom: 100%; left: 0; background: #333; color: white; padding: 8px; border-radius: 4px; white-space: nowrap; z-index: 1000; font-size: 0.9em; }}
        .substitute-text:hover .origin-tooltip {{ display: block; }}
        .indent-1 {{ margin-left: 20px; }}
        .indent-2 {{ margin-left: 40px; }}
        .indent-3 {{ margin-left: 60px; }}
        .indent-4 {{ margin-left: 80px; }}
        .indent-5 {{ margin-left: 100px; }}
    </style>
    <script>
        function toggleNode(id) {{
            const el = document.getElementById(id);
            el.classList.toggle('visible');
        }}
    </script>
</head>
<body>
    <h1>Parse Tree: "{phrase}"</h1>
    <div class="tree">
"""

    # NEW: Render parse tree recursively from cache (SIMPLIFIED - no expensive lookups)
    def render_parse_from_cache_simple(span_text, split, cache, depth=0):
        indent_class = f"indent-{min(depth, 5)}"

        # Get this parse from cache
        cache_key = (span_text, split)
        if cache_key not in cache:
            # Just show as leaf
            return f'<div class="{indent_class}"><span class="span-text">{span_text}</span></div>'

        cache_value = cache[cache_key]
        energy = cache_value[3]
        num_subs = len(cache_value[0])

        # Simple display - just text, energy, and substitute count
        html_parts = []
        html_parts.append(f'<div class="{indent_class}">')
        html_parts.append(f'<span class="span-text">{span_text}</span> ')
        html_parts.append(f'<span class="energy">E={energy:.2f}</span> ')
        html_parts.append(f'<span class="score">({num_subs} subs)</span> ')

        # Substitutes section - merged from all parses
        html_parts.append(f'<div class="substitutes" id="{node_id}">')
        merged_subs = get_all_substitutes_merged(cache, span_text)

        if merged_subs:
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            html_parts.append(f'<strong>Substitutes ({len(merged_subs)}): ')
            html_parts.append(f'<span class="sub-aggregation">{agg_count} aggregation</span>, ')
            html_parts.append(f'<span class="sub-expansion">{exp_count} expansion</span></strong>')

            for sub_text, score, source_type, sub_origins in merged_subs[:50]:  # Limit to 50 for display
                sub_pattern = catalog.get_unit(sub_text)
                if sub_pattern:
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(5)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(5)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    origin_html = ''
                    if sub_origins:
                        left_path = f'"{sub_origins["left_source"]}" → "{sub_origins["left_sub"]}" ({sub_origins["left_score"]:.2f})'
                        right_path = f'"{sub_origins["right_source"]}" → "{sub_origins["right_sub"]}" ({sub_origins["right_score"]:.2f})'
                        origin_html = f'<span class="origin-tooltip">Aggregation: {left_path} + {right_path}</span>'
                    elif source_type == 'expansion':
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context</span>'

                    source_class = f'sub-{source_type}'
                    html_parts.append(f'<div class="sub-item">')
                    html_parts.append(f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | ')
                    html_parts.append(f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> ')
                    html_parts.append(f'<span class="score">(score: {score:.4f})</span> | ')
                    html_parts.append(f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>')
                    html_parts.append(f'</div>')
        html_parts.append('</div>')

        # Recursively render children if this is a binary split
        # Add depth limit to prevent infinite recursion
        if split and isinstance(split, tuple) and len(split) == 2 and depth < 20:
            # Handle special split types (aggregation, expansion)
            if split[0] in ['aggregation', 'expansion']:
                # For these, split[1] is a tuple of all tokens, not a binary split
                # Try to find a binary split by checking the cache for actual splits
                tokens_tuple = split[1]
                if len(tokens_tuple) == 2:
                    # Binary case - can render directly
                    left_text, right_text = tokens_tuple
                else:
                    # Multi-word case - find best binary split by checking cache
                    # Look for a split of this span in the cache
                    best_binary_split = None
                    for (cached_text, cached_split), _ in cache.items():
                        if cached_text == span_text and cached_split != split:
                            if isinstance(cached_split, tuple) and len(cached_split) == 2:
                                if cached_split[0] not in ['aggregation', 'expansion']:
                                    best_binary_split = cached_split
                                    break

                    if best_binary_split:
                        left_text, right_text = best_binary_split
                    else:
                        # No binary split found, just show tokens as leaves
                        for token in tokens_tuple:
                            html_parts.append(f'<div class="indent-{min(depth+1, 5)}"><span class="span-text">{token}</span></div>')
                        html_parts.append('</div>')
                        return ''.join(html_parts)
            else:
                # Regular binary split
                left_text, right_text = split

            # Only recurse if children are multi-word (not leaves)
            if ' ' in left_text:
                # Find best split for left child
                _, _, _, _, left_split, _ = get_best_parse(cache, left_text)
                html_parts.append(render_parse_from_cache(left_text, left_split, cache, catalog, depth + 1))
            else:
                # Leaf node - just show the word
                html_parts.append(f'<div class="indent-{min(depth+1, 5)}"><span class="span-text">{left_text}</span></div>')

            if ' ' in right_text:
                # Find best split for right child
                _, _, _, _, right_split, _ = get_best_parse(cache, right_text)
                html_parts.append(render_parse_from_cache(right_text, right_split, cache, catalog, depth + 1))
            else:
                # Leaf node - just show the word
                html_parts.append(f'<div class="indent-{min(depth+1, 5)}"><span class="span-text">{right_text}</span></div>')

        html_parts.append('</div>')
        return ''.join(html_parts)

    # Helper function to create consistent IDs from span text
    def make_span_id(span_text):
        return f"subs-{span_text.replace(' ', '-')}"

    # Generate tree HTML recursively
    def render_node(node, depth=0):
        indent_class = f"indent-{min(depth, 5)}"
        span_text = node.span.text

        # Get best parse for this span
        subs, _, _, energy, split, origins = get_best_parse(subparse_cache, span_text)

        node_id = make_span_id(span_text)

        html_parts = []
        html_parts.append(f'<div class="{indent_class}">')
        html_parts.append(f'<div class="node" onclick="toggleNode(\'{node_id}\')">')
        html_parts.append(f'<span class="span-text">{span_text}</span> ')
        html_parts.append(f'<span class="energy">E={energy:.2f}</span> ')
        if split:
            split_str = str(split) if isinstance(split, tuple) and split[0] != 'expansion' else 'expansion'
            html_parts.append(f'<span class="split-info">split: {split_str}</span>')
        html_parts.append('</div>')

        # Substitutes section (hidden by default) - MERGED from all parses
        html_parts.append(f'<div class="substitutes" id="{node_id}">')

        # Get ALL substitutes from all parses, merged
        merged_subs = get_all_substitutes_merged(subparse_cache, span_text)

        if merged_subs:
            # Count by source type
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            html_parts.append(f'<strong>Substitutes ({len(merged_subs)}): ')
            html_parts.append(f'<span class="sub-aggregation">{agg_count} aggregation</span>, ')
            html_parts.append(f'<span class="sub-expansion">{exp_count} expansion</span></strong>')

            # Show ALL merged substitutes (no limit) since they're hidden until clicked
            for sub_text, score, source_type, sub_origins in merged_subs:
                # Get contexts for this substitute from catalog
                sub_pattern = catalog.get_unit(sub_text)
                if sub_pattern:
                    # Get top contexts (most frequent)
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(10)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(10)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    # Build origin tooltip based on source type
                    origin_html = ''
                    if sub_origins:
                        # Aggregation: show combination origin
                        left_path = f'"{sub_origins["left_source"]}" → "{sub_origins["left_sub"]}" ({sub_origins["left_score"]:.2f})'
                        right_path = f'"{sub_origins["right_source"]}" → "{sub_origins["right_sub"]}" ({sub_origins["right_score"]:.2f})'
                        origin_html = f'<span class="origin-tooltip">Aggregation: {left_path} + {right_path}</span>'
                    elif source_type == 'expansion':
                        # Expansion: show it came from parent context
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context mutual expansion</span>'

                    # Color code by source type
                    source_class = f'sub-{source_type}'

                    html_parts.append(f'<div class="sub-item">')
                    html_parts.append(f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | ')
                    html_parts.append(f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> ')
                    html_parts.append(f'<span class="score">(score: {score:.4f})</span> | ')
                    html_parts.append(f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>')
                    html_parts.append(f'</div>')
                else:
                    # No pattern in catalog, just show substitute
                    source_class = f'sub-{source_type}'
                    html_parts.append(f'<div class="sub-item"><span class="{source_class}">{sub_text}</span> <span class="score">(score: {score:.4f})</span></div>')
        else:
            html_parts.append('<em>No substitutes</em>')
        html_parts.append('</div>')

        # NEW: Show all alternative parses for this span
        all_parses = [(k, v) for k, v in subparse_cache.items() if k[0] == span_text]
        if len(all_parses) > 1:
            # Sort by energy (best first)
            all_parses_sorted = sorted(all_parses, key=lambda x: x[1][3])

            html_parts.append(f'<div class="alternatives" style="margin-top: 10px; padding: 10px; background: #fff3e0; border-radius: 4px;">')
            html_parts.append(f'<strong>Alternative parses ({len(all_parses)}):</strong>')

            for parse_key, parse_value in all_parses_sorted:
                _, parse_split = parse_key
                parse_subs, _, _, parse_energy = parse_value[:4]
                parse_origins = parse_value[4] if len(parse_value) == 5 else None

                is_current = (parse_split == split)
                style = 'font-weight: bold; color: #2196F3;' if is_current else ''

                split_str = str(parse_split) if isinstance(parse_split, tuple) else 'None'
                num_multiword = len([s for s, _ in parse_subs if ' ' in s])

                html_parts.append(f'<div style="margin: 5px 0; padding: 5px; background: white; border-radius: 3px; {style}">')
                html_parts.append(f'{"→ " if is_current else "  "}Split: {split_str} | ')
                html_parts.append(f'Energy: {parse_energy:.2f} | ')
                html_parts.append(f'Subs: {len(parse_subs)} ({num_multiword} multi-word)')
                if is_current:
                    html_parts.append(f' <em>(current)</em>')
                html_parts.append(f'</div>')

            html_parts.append('</div>')

        # Render children
        if node.left:
            html_parts.append(render_node(node.left, depth + 1))
        if node.right:
            html_parts.append(render_node(node.right, depth + 1))

        html_parts.append('</div>')
        return ''.join(html_parts)

    # Render alternative trees (simple structure, no expensive lookups)
    print(f"Adding {len(parse_buffer)} alternative parse trees...")

    # Find the BEST split for a span (lowest energy = winning parse)
    def find_best_split(span_text):
        best_split = None
        best_energy = float('inf')
        for text, split, energy, _ in parse_buffer:
            if text == span_text and energy < best_energy:
                best_split = split
                best_energy = energy
        return best_split

    # Render tree in bracket notation: (my (hat again))
    def render_as_brackets(span_text, split):
        if not split:
            return span_text

        if isinstance(split, tuple) and len(split) == 2:
            if split[0] in ['aggregation', 'expansion']:
                tokens = split[1]
                if len(tokens) == 2:
                    left_text, right_text = tokens
                else:
                    return f"({' '.join(tokens)})"
            else:
                left_text, right_text = split

            # Recursively render children with WINNING parse structure
            if ' ' in left_text:
                left_split = find_best_split(left_text)
                left_repr = render_as_brackets(left_text, left_split)
            else:
                left_repr = left_text

            if ' ' in right_text:
                right_split = find_best_split(right_text)
                right_repr = render_as_brackets(right_text, right_split)
            else:
                right_repr = right_text

            return f"({left_repr} {right_repr})"

        return span_text

    # Render a leaf node (single word) with substitutes
    def render_leaf(word_text, depth, context_side=None, context_word=None):
        """
        Render a single-word leaf.

        Args:
            word_text: The word to render
            depth: Indentation depth
            context_side: 'left' or 'right' indicating which side of parent split
            context_word: The sibling word that provides context
        """
        indent_class = f"indent-{min(depth, 5)}"
        span_id = make_span_id(word_text) + f'-leaf-{depth}'

        result = f'<div class="{indent_class}">'
        result += f'<span class="span-text" onclick="toggleNode(\'{span_id}\')" style="cursor: pointer;">{word_text}</span>'

        # Add substitutes section for single word
        result += f'<div class="substitutes" id="{span_id}">'
        merged_subs = get_all_substitutes_merged(subparse_cache, word_text)
        if merged_subs:
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            result += f'<strong>Substitutes ({len(merged_subs)}): '
            result += f'<span class="sub-aggregation">{agg_count} aggregation</span>, '
            result += f'<span class="sub-expansion">{exp_count} expansion</span></strong>'

            for sub_text, score, source_type, sub_origins in merged_subs:  # Show all
                sub_pattern = catalog.get_unit(sub_text)
                if sub_pattern:
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(5)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(5)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    # Build origin tooltip (special handling for leaves with context info)
                    origin_html = ''
                    if context_word and source_type == 'expansion':
                        # For leaf nodes, show which word provided the context
                        side_label = 'Right' if context_side == 'right' else 'Left'
                        origin_html = f'<span class="origin-tooltip">{side_label} context of ({context_word})</span>'
                    elif sub_origins:
                        if 'left_sub' in sub_origins and 'right_sub' in sub_origins:
                            # Detailed aggregation: show which substitutes were combined and their context sources
                            left_sub = sub_origins['left_sub']
                            right_sub = sub_origins['right_sub']
                            left_source = sub_origins['left_source']
                            right_source = sub_origins['right_source']
                            origin_html = f'<span class="origin-tooltip">Aggregate from "{left_sub}" left context of ({left_source}) and "{right_sub}" right context of ({right_source})</span>'
                        elif sub_origins.get('split_type') == 'expansion':
                            # Expansion: show parent split
                            parent = sub_origins.get('parent_split', 'unknown')
                            origin_html = f'<span class="origin-tooltip">Expansion: from parent context {parent}</span>'
                        elif sub_origins.get('split_type') == 'aggregation':
                            # Simple aggregation: show the split
                            parent = sub_origins.get('parent_split', 'unknown')
                            origin_html = f'<span class="origin-tooltip">Aggregation: from constituent combination {parent}</span>'
                    elif source_type == 'expansion':
                        # Fallback expansion tooltip
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context mutual expansion</span>'

                    source_class = f'sub-{source_type}'
                    result += f'<div class="sub-item">'
                    result += f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | '
                    result += f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> '
                    result += f'<span class="score">(score: {score:.4f})</span> | '
                    result += f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>'
                    result += f'</div>'
        else:
            result += '<em>No substitutes</em>'
        result += '</div></div>'
        return result

    # Simple tree renderer - no cache scans, just uses the split info!
    def render_simple_tree(span_text, split, depth=0):
        indent_class = f"indent-{min(depth, 5)}"

        # Get basic info if in cache
        cache_key = (span_text, split)
        if cache_key in subparse_cache:
            energy = subparse_cache[cache_key][3]
            num_subs = len(subparse_cache[cache_key][0])
            info = f'E={energy:.2f}, {num_subs} subs'
        else:
            info = ''

        # Create unique ID for this node's substitutes
        span_id = make_span_id(span_text) + f'-alt-{depth}-{id(split)}'

        result = f'<div class="{indent_class}">'
        result += f'<span class="span-text" onclick="toggleNode(\'{span_id}\')" style="cursor: pointer;">{span_text}</span>'
        if info:
            result += f' <span class="energy">{info}</span>'

        # Add substitutes section (hidden by default)
        result += f'<div class="substitutes" id="{span_id}">'
        merged_subs = get_all_substitutes_merged(subparse_cache, span_text)
        if merged_subs:
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            result += f'<strong>Substitutes ({len(merged_subs)}): '
            result += f'<span class="sub-aggregation">{agg_count} aggregation</span>, '
            result += f'<span class="sub-expansion">{exp_count} expansion</span></strong>'

            for sub_text, score, source_type, sub_origins in merged_subs:  # Show all
                sub_pattern = catalog.get_unit(sub_text)
                if sub_pattern:
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(5)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(5)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    # Build origin tooltip
                    origin_html = ''
                    if sub_origins:
                        if 'left_source' in sub_origins:
                            # Detailed aggregation: show combination origin
                            left_path = f'"{sub_origins["left_source"]}" → "{sub_origins["left_sub"]}" ({sub_origins["left_score"]:.2f})'
                            right_path = f'"{sub_origins["right_source"]}" → "{sub_origins["right_sub"]}" ({sub_origins["right_score"]:.2f})'
                            origin_html = f'<span class="origin-tooltip">Aggregation: {left_path} + {right_path}</span>'
                        elif sub_origins.get('split_type') == 'expansion':
                            # Expansion: show parent split
                            parent = sub_origins.get('parent_split', 'unknown')
                            origin_html = f'<span class="origin-tooltip">Expansion: from parent context {parent}</span>'
                        elif sub_origins.get('split_type') == 'aggregation':
                            # Simple aggregation: show the split
                            parent = sub_origins.get('parent_split', 'unknown')
                            origin_html = f'<span class="origin-tooltip">Aggregation: from constituent combination {parent}</span>'
                    elif source_type == 'expansion':
                        # Fallback expansion tooltip
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context mutual expansion</span>'

                    source_class = f'sub-{source_type}'
                    result += f'<div class="sub-item">'
                    result += f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | '
                    result += f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> '
                    result += f'<span class="score">(score: {score:.4f})</span> | '
                    result += f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>'
                    result += f'</div>'
        else:
            result += '<em>No substitutes</em>'
        result += '</div>'

        # Recurse on children (split tells us what they are!)
        if split and isinstance(split, tuple) and len(split) == 2 and depth < 10:
            if split[0] in ['aggregation', 'expansion']:
                tokens = split[1]
                if len(tokens) == 2:
                    left_text, right_text = tokens
                else:
                    result += f' <em>[{len(tokens)} tokens]</em></div>'
                    return result
            else:
                left_text, right_text = split

            # Recurse - for multi-word children, use their WINNING split
            if ' ' in left_text:
                left_split = find_best_split(left_text)
                result += render_simple_tree(left_text, left_split, depth + 1)
            else:
                # Left child: substitutes come from left_words of right sibling
                result += render_leaf(left_text, depth + 1, context_side='left', context_word=right_text)

            if ' ' in right_text:
                right_split = find_best_split(right_text)
                result += render_simple_tree(right_text, right_split, depth + 1)
            else:
                # Right child: substitutes come from right_words of left sibling
                result += render_leaf(right_text, depth + 1, context_side='right', context_word=left_text)

        result += '</div>'
        return result

    # Render alternative trees in processing order
    html += '<h2 style="margin-top: 40px;">Alternative Parse Trees (incremental processing order)</h2>'

    all_parses = [(span_length, span_text, split, energy)
                  for span_text, split, energy, span_length in parse_buffer]
    all_parses.sort(key=lambda x: (x[0], x[3]))  # Sort by length, then energy

    current_length = 0
    for span_length, span_text, split, energy in all_parses:
        if span_length < 2:
            continue

        if span_length != current_length:
            if current_length > 0:
                html += '</div>'
            html += f'<h3 style="margin-top: 20px; color: #666;">{span_length}-word spans</h3>'
            html += '<div style="margin-left: 20px;">'
            current_length = span_length

        # Show in bracket notation: (my (hat again))
        bracket_repr = render_as_brackets(span_text, split)
        html += f'<div class="tree" style="margin: 10px 0; padding: 10px; background: #fafafa; border-left: 3px solid #2196F3;">'
        html += f'<strong style="font-family: monospace; font-size: 1.1em;">{bracket_repr}</strong>'

        # Get energy info
        cache_key = (span_text, split)
        if cache_key in subparse_cache:
            energy = subparse_cache[cache_key][3]
            num_subs = len(subparse_cache[cache_key][0])
            html += f' <span class="energy">E={energy:.2f}, {num_subs} subs</span>'
        html += '<br>'

        html += render_simple_tree(span_text, split, depth=0)
        html += '</div>'

    if current_length > 0:
        html += '</div>'

    html += """
    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nInteractive parse tree saved to: {output_file}")
    print(f"Open it in your browser to explore the tree structure!")


def main():
    import argparse
    parser_args = argparse.ArgumentParser(
        description='Analyze a phrase: parse it and show contextual substitution classes.'
    )
    parser_args.add_argument('phrase', help='Phrase to analyze')
    parser_args.add_argument('--scoring', '-s',
                            choices=['cosine', 'ic_cosine', 'pmi'],
                            default='cosine',
                            help='Scoring method: cosine (default), ic_cosine, or pmi')
    parser_args.add_argument('--compare', '-c', action='store_true',
                            help='Compare all three scoring methods')
    args = parser_args.parse_args()

    # Load models
    print("Loading models...")
    catalog = UnitCatalog()
    catalog.load('unit_catalog.pkl')
    catalog.build_gpu_index(min_freq=1)  # Index all units, maximize information
    parser = IncrementalBidirParser(catalog, debug=False)

    # Analyze
    if args.compare:
        for scoring in ['cosine', 'ic_cosine', 'pmi']:
            tree, cache, parse_buffer = analyze(args.phrase, catalog, parser, scoring=scoring)
            print("\n" + "=" * 70 + "\n")
        # Export HTML for last scoring method
        export_html_tree(args.phrase, tree, cache, catalog, parse_buffer, f"parse_tree_{args.phrase.replace(' ', '_')}.html")
    else:
        tree, cache, parse_buffer = analyze(args.phrase, catalog, parser, scoring=args.scoring)
        # Export HTML
        export_html_tree(args.phrase, tree, cache, catalog, parse_buffer, f"parse_tree_{args.phrase.replace(' ', '_')}.html")

if __name__ == "__main__":
    main()
