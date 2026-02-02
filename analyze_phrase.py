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

from bidir_simple import UnitCatalog, SimpleBidirParser, print_tree


def analyze(phrase: str, catalog: UnitCatalog, parser: SimpleBidirParser,
            max_class_members: int = 5):
    """Analyze a phrase and display results."""

    tokens = phrase.lower().split()

    print("=" * 70)
    print(f'INPUT: "{phrase}"')
    print("=" * 70)

    # 1. Parse the phrase
    print("\n[1] PARSE TREE")
    print("-" * 40)
    if len(tokens) >= 2:
        tree = parser.parse(tokens)
        print_tree(tree)
    else:
        print(f'  "{phrase}" (single token)')
        tree = None

    # 2. Show contextual substitution classes for each node
    print("\n[2] CONTEXTUAL SUBSTITUTION CLASSES")
    print("-" * 40)
    print("(Units that share presented context, scored by similarity to target)")

    if tree:
        show_contextual_expansions(tree, tokens, catalog, max_class_members)
    else:
        # Single token
        show_single_token_expansion(tokens[0], catalog, max_class_members)

    print()


def show_contextual_expansions(node, full_tokens, catalog, max_members, context_window=3):
    """Recursively show contextual expansions for each node."""
    span = node.span

    # Reconstruct the context that was used during parsing
    left_ctx = full_tokens[max(0, span.start - context_window):span.start]
    right_ctx = full_tokens[span.end:min(len(full_tokens), span.end + context_window)]

    # Get contextual candidates
    candidates = catalog.gpu_contextual_candidates(
        span.text,
        left_context=left_ctx,
        right_context=right_ctx,
        max_results=max_members + 1
    )

    # Filter out the span itself
    candidates = [(text, score) for text, score in candidates if text != span.text][:max_members]

    # Display
    ctx_str = f'[{" ".join(left_ctx)}] ___ [{" ".join(right_ctx)}]' if left_ctx or right_ctx else "(no context)"
    print(f'\n  "{span.text}"  in context  {ctx_str}')

    if candidates:
        print(f'    Expansion size: {node.unit_expansion}')
        print(f'    Top substitutes:')
        for text, score in candidates:
            print(f'      {score:.3f}  "{text}"')
    else:
        if node.unit_expansion > 1:
            print(f'    Expansion size: {node.unit_expansion} (candidates not retrieved)')
        else:
            print(f'    (no contextual substitutes found)')

    # Recurse to children
    if node.left:
        show_contextual_expansions(node.left, full_tokens, catalog, max_members, context_window)
    if node.right:
        show_contextual_expansions(node.right, full_tokens, catalog, max_members, context_window)


def show_single_token_expansion(token, catalog, max_members):
    """Show expansion for a single token (no parse tree context)."""
    # For single tokens, we have no sibling context
    candidates = catalog.gpu_contextual_candidates(
        token,
        left_context=[],
        right_context=[],
        max_results=max_members + 1
    )
    candidates = [(text, score) for text, score in candidates if text != token][:max_members]

    print(f'\n  "{token}"  (single token, no context)')
    if candidates:
        for text, score in candidates:
            print(f'    {score:.3f}  "{text}"')
    else:
        print(f'    (no substitutes found)')


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phrase.py \"phrase to analyze\"")
        print("\nExamples:")
        print('  python analyze_phrase.py "we should go"')
        print('  python analyze_phrase.py "that\'s a beautiful hat"')
        print('  python analyze_phrase.py "i want to see you"')
        sys.exit(1)

    phrase = sys.argv[1]

    # Load models
    print("Loading models...")
    catalog = UnitCatalog()
    catalog.load('unit_catalog.pkl')
    catalog.build_gpu_index(min_freq=10)

    parser = SimpleBidirParser(catalog, debug=False)

    # Analyze
    analyze(phrase, catalog, parser)


if __name__ == "__main__":
    main()
