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

def analyze(phrase: str, catalog: UnitCatalog, parser: IncrementalBidirParser,
            max_class_members: int = 10, scoring: str = 'cosine', external_left: list = [], external_right: list = []):
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
    for k, word in enumerate(tokens, 1):
        parser.add_word(word)  # Add word sequentially
        prefix_tokens = tokens[:k]
        prefix = " ".join(prefix_tokens)
        print(f"\nAdding word '{word}' → Prefix: \"{prefix}\"")

        # Bootstrap single word
        if k == 1:
            subs = catalog.gpu_contextual_candidates(prefix, external_left, external_right, max_class_members + 1, scoring)
            subs = [(t, s) for t, s in subs if t != prefix][:max_class_members]
            pattern = catalog.get_unit(prefix)
            eff_left = [w for w, c in pattern.left_words.most_common(20) if c >= 5] if pattern else []
            eff_right = [w for w, c in pattern.right_words.most_common(20) if c >= 5] if pattern else []
            subparse_cache[prefix] = (subs, eff_left, eff_right)
            print(f"  Bootstrap subs for \"{prefix}\": {subs}")
            continue

        # Test all pairwise splits (binary mappings), building backwards with reuse
        for split in range(1, k):  # Split points (left=1..k-1, right=rest)
            left_text = " ".join(prefix_tokens[:split])
            right_text = " ".join(prefix_tokens[split:])
            print(f"  Testing pairwise mapping: (\"{left_text}\" \"{right_text}\")")

            # Get cached parses for left and right
            left_subs, left_eff_left, left_eff_right = subparse_cache.get(left_text, ([], [], []))
            right_subs, right_eff_left, right_eff_right = subparse_cache.get(right_text, ([], [], []))

            # Bootstrap right if not cached yet
            if not right_subs:
                pattern = catalog.get_unit(right_text)
                if pattern:
                    right_eff_left = [w for w, c in pattern.left_words.most_common(20) if c >= 5]
                    right_eff_right = [w for w, c in pattern.right_words.most_common(20) if c >= 5]
                else:
                    right_eff_left = []
                    right_eff_right = []
                # For initial bootstrap, right_subs is just the unit itself
                right_subs = [(right_text, 1.0)]
                subparse_cache[right_text] = (right_subs, right_eff_left, right_eff_right)

            # MUTUAL EXPANSION: Only proceed if we have context from BOTH sides
            # This ensures we have actual fillers to seed the expansion

            print(f"    Left context available: {len(left_eff_left)} left, {len(left_eff_right)} right")
            print(f"    Right context available: {len(right_eff_left)} left, {len(right_eff_right)} right")

            # Skip mutual expansion if either side has no context fillers
            if not left_eff_right and not right_eff_left:
                print(f"    Skipping mutual expansion: no context fillers from either side yet")
                # Cache the prefix with just the units themselves
                subparse_cache[prefix] = ([(left_text, 1.0), (right_text, 1.0)], left_eff_left, left_eff_right)
                continue

            # Step 1: Get substitutes for LEFT side using right's left context
            print(f"    [STEP 1] Get substitutes for LEFT: \"{left_text}\"")
            left_subs_for_expansion = catalog.gpu_contextual_candidates(
                left_text,
                left_context=external_left,
                right_context=list(right_eff_left) + external_right if right_eff_left else external_right,
                max_results=max_class_members * 3,
                scoring=scoring
            )
            left_subs_for_expansion = [(t, s) for t, s in left_subs_for_expansion if t != left_text][:max_class_members]
            print(f"      Found {len(left_subs_for_expansion)} substitutes for left")

            # Collect right_words from left substitutes → candidates for RIGHT
            left_right_contexts = set()
            for sub_text, _ in left_subs_for_expansion:
                sub_pattern = catalog.get_unit(sub_text)
                if sub_pattern:
                    for word in sub_pattern.right_words.keys():
                        left_right_contexts.add(word)
            print(f"      Collected {len(left_right_contexts)} right-context words from left subs")

            # Step 2: Get substitutes for RIGHT side using left's right context
            print(f"    [STEP 2] Get substitutes for RIGHT: \"{right_text}\"")
            right_subs_for_expansion = catalog.gpu_contextual_candidates(
                right_text,
                left_context=list(left_right_contexts) + external_left if left_right_contexts else external_left,
                right_context=external_right,
                max_results=max_class_members * 3,
                scoring=scoring
            )
            right_subs_for_expansion = [(t, s) for t, s in right_subs_for_expansion if t != right_text][:max_class_members]
            print(f"      Found {len(right_subs_for_expansion)} substitutes for right")

            # Collect left_words from right substitutes → candidates for LEFT
            right_left_contexts = set()
            for sub_text, _ in right_subs_for_expansion:
                sub_pattern = catalog.get_unit(sub_text)
                if sub_pattern:
                    for word in sub_pattern.left_words.keys():
                        right_left_contexts.add(word)
            print(f"      Collected {len(right_left_contexts)} left-context words from right subs")

            # Step 3: Score candidates for RIGHT using left's right_contexts as seed
            print(f"    [STEP 3] Score RIGHT candidates using left's right contexts")
            right_candidates = catalog.gpu_contextual_candidates(
                right_text,
                left_context=list(left_right_contexts) + external_left if left_right_contexts else external_left,
                right_context=external_right,
                max_results=max_class_members + 1,
                scoring=scoring
            )
            right_candidates = [(t, s) for t, s in right_candidates if t != right_text][:max_class_members]
            print(f"      Scored right candidates: {len(right_candidates)} results")

            # Step 4: Score candidates for LEFT using right's left_contexts as seed
            print(f"    [STEP 4] Score LEFT candidates using right's left contexts")
            left_candidates = catalog.gpu_contextual_candidates(
                left_text,
                left_context=external_left,
                right_context=list(right_left_contexts) + external_right if right_left_contexts else external_right,
                max_results=max_class_members + 1,
                scoring=scoring
            )
            left_candidates = [(t, s) for t, s in left_candidates if t != left_text][:max_class_members]
            print(f"      Scored left candidates: {len(left_candidates)} results")

            print(f"    Left candidates: {left_candidates[:5]}")
            print(f"    Right candidates: {right_candidates[:5]}")

            # Update cache for whole prefix
            # Combine left and right candidates as substitutes for the full span
            combined_candidates = []

            # Add left candidates (they replace the whole left side)
            for t, s in left_candidates:
                combined_candidates.append((t, s))

            # Add right candidates (they replace the whole right side)
            for t, s in right_candidates:
                combined_candidates.append((t, s))

            # Sort by score and keep top
            combined_candidates = sorted(set(combined_candidates), key=lambda x: -x[1])[:max_class_members]

            # Update context words from both sides
            eff_left = list(set(left_eff_left) | set(right_eff_left))
            eff_right = list(set(left_eff_right) | set(right_eff_right))

            subparse_cache[prefix] = (combined_candidates, eff_left, eff_right)

            print(f"    Cache updated for \"{prefix}\": {len(combined_candidates)} combined candidates")

    tree = parser.get_current_parse()

    # Final sections (original)
    print("\n[2] FINAL PARSE TREE")
    print("-" * 40)
    if tree:
        print_tree(tree)
    else:
        print(f' "{phrase}" (single token)')

    print("\n[3] CONTEXTUAL SUBSTITUTION CLASSES")
    print("-" * 40)

    # Display substitution classes for nodes in the final parse tree
    if tree:
        def show_node_subs(node, depth=0):
            """Recursively show substitution classes for each node in parse tree."""
            indent = "  " * depth
            span_text = node.span.text

            # Get substitutes for this span using external context only
            subs = catalog.gpu_contextual_candidates(
                span_text,
                left_context=external_left,
                right_context=external_right,
                max_results=5,
                scoring=scoring
            )
            subs = [(t, s) for t, s in subs if t != span_text][:5]

            print(f"{indent}Span: \"{span_text}\"")
            if subs:
                print(f"{indent}  Substitution class:")
                for sub_text, score in subs:
                    print(f"{indent}    - {sub_text:.20s} (score: {score:.4f})")
            else:
                print(f"{indent}  (No substitutes found)")

            # Recursively show left and right children
            if node.left:
                show_node_subs(node.left, depth + 1)
            if node.right:
                show_node_subs(node.right, depth + 1)

        show_node_subs(tree)
    else:
        print("No parse tree to display")

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
    catalog.build_gpu_index(min_freq=10)
    parser = IncrementalBidirParser(catalog, debug=False)

    # Analyze
    if args.compare:
        for scoring in ['cosine', 'ic_cosine', 'pmi']:
            analyze(args.phrase, catalog, parser, scoring=scoring)
            print("\n" + "=" * 70 + "\n")
    else:
        analyze(args.phrase, catalog, parser, scoring=args.scoring)

if __name__ == "__main__":
    main()
