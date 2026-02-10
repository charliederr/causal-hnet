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

            # Reuse from cache (subparse)
            left_subs, left_eff_left, left_eff_right = subparse_cache.get(left_text, ([], [], []))

            # Reuse or bootstrap right
            right_subs, right_eff_left, right_eff_right = subparse_cache.get(right_text, ([], [], []))
            if not right_subs:
                right_subs = catalog.gpu_contextual_candidates(right_text, external_left, external_right, max_class_members + 1, scoring)
                right_subs = [(t, s) for t, s in right_subs if t != right_text][:max_class_members]
                pattern = catalog.get_unit(right_text)
                right_eff_left = [w for w, c in pattern.left_words.most_common(20) if c >= 5] if pattern else []
                right_eff_right = [w for w, c in pattern.right_words.most_common(20) if c >= 5] if pattern else []
                subparse_cache[right_text] = (right_subs, right_eff_left, right_eff_right)

            # Step 1: Generate substitutes for right seeded by right contexts of left (as per your goal)
            print(f"    [DEBUG] Seeding right subs with right contexts of left: {left_eff_right[:10]} ... (total {len(left_eff_right)})")
#            right_candidates = catalog.gpu_contextual_candidates(
#                right_text,
#                left_context=[], 
#                right_context=left_eff_right + external_right,  # Seed with right of left
#                max_results=max_class_members * 2,
#                scoring=scoring,
#                seed_side='right'  # Use the new param to seed only right
#            )
#            right_candidates = sorted([(t, s) for t, s in right_candidates if t != right_text], key=lambda x: -x[1])[:max_class_members]

            # Step 2: Generate substitutes for left seeded by left contexts of right
            print(f"    [DEBUG] Seeding left subs with left contexts of right: {right_eff_left[:10]} ... (total {len(right_eff_left)})")
#            left_candidates = catalog.gpu_contextual_candidates(
#                left_text,
#                left_context=right_eff_left + external_left,  # Seed with left of right
#                right_context=[], 
#                max_results=max_class_members * 2,
#                scoring=scoring,
#                seed_side='left'  # Seed only left
#            )
#            left_candidates = sorted([(t, s) for t, s in left_candidates if t != left_text], key=lambda x: -x[1])[:max_class_members]
#
#            print(f"    Initial substitutes for right: {right_candidates}")
#            print(f"    Initial substitutes for left: {left_candidates}")

            # Step 3: Mutual filter with both sides (filter candidates using contexts from both)

            refined_right = catalog.gpu_contextual_candidates(
                left_text, # source of "seed" substitutes for right hand side
                left_context=right_eff_left + external_left,     # own left
                right_context=right_eff_right + external_right,  # own right
                max_results=max_class_members + 1,
                scoring=scoring
            )
            refined_right = sorted([(t, s) for t, s in refined_right if t != left_text], key=lambda x: -x[1])[:max_class_members]

            refined_left = catalog.gpu_contextual_candidates(
                right_text, # source of "seed" substitutes for left hand side
                left_context=left_eff_left + external_left,    # own left
                right_context=left_eff_right + external_right, # own right
                max_results=max_class_members + 1,
                scoring=scoring
            )
            refined_left = sorted([(t, s) for t, s in refined_left if t != right_text], key=lambda x: -x[1])[:max_class_members]

            print(f"    Refined right subs (filtered both sides): {refined_right}")
            print(f"    Refined left subs (filtered both sides): {refined_left}")

            # Final mutual set: ranked by combined score (average from both sides for matching texts)
            mutual = {}
            for t, s_left in refined_left:
                for cand_t, s_right in refined_right:
                    if t == cand_t:
                        mutual[t] = (s_left + s_right) / 2
            mutual_sorted = sorted(mutual.items(), key=lambda x: -x[1])[:max_class_members]
            print(f"    Final mutual set (combined scores): {mutual_sorted}")

            # Update cache for whole prefix (use mutual as subs, combined eff)
            eff_left = list(set(left_eff_left) | set(right_eff_left))
            eff_right = list(set(left_eff_right) | set(right_eff_right))
            subparse_cache[prefix] = (mutual_sorted, eff_left, eff_right)

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
    # (keep your show_contextual_expansions / show_single_token_expansion; add external ctx to calls if needed)

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
