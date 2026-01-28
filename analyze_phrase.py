#!/usr/bin/env python3
"""
Analyze a phrase: parse it, show substitution classes for sub-spans, and predict continuations.

Usage:
    python analyze_phrase.py "we should go"
    python analyze_phrase.py "i want to"
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from bidir_simple import UnitCatalog, SimpleBidirParser, ContextPattern, print_tree
from substitution_predictor import SubstitutionPredictor


def analyze(phrase: str, predictor: SubstitutionPredictor, parser: SimpleBidirParser,
            max_class_members: int = 5, max_predictions: int = 10):
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

    # 2. Collect all spans from the parse tree
    print("\n[2] SUBSTITUTION CLASSES FOR SUB-SPANS")
    print("-" * 40)

    if len(tokens) >= 2:
        spans = collect_spans(tree)
        # Sort by span length (longest first) then by position
        spans.sort(key=lambda x: (-len(x.split()), phrase.find(x)))

        for span_text in spans:
            sub_class = predictor.find_substitution_class(span_text, max_members=max_class_members + 1)
            if sub_class:
                # Filter out the span itself
                members = [(m, s) for m, s in sub_class if m != span_text][:max_class_members]
                if members:
                    print(f'\n  "{span_text}" ≈')
                    for member, score in members:
                        print(f'    {score:.3f}  "{member}"')
            else:
                # Check if it's in catalog but has no matches
                if span_text in predictor.unit_patterns:
                    print(f'\n  "{span_text}" ≈ (no similar units found)')
    else:
        sub_class = predictor.find_substitution_class(phrase.lower(), max_members=max_class_members)
        if sub_class:
            print(f'\n  "{phrase}" ≈')
            for member, score in sub_class:
                print(f'    {score:.3f}  "{member}"')
        else:
            print(f'  "{phrase}" not found in catalog')

    # 3. Predictions
    print("\n[3] PREDICTED CONTINUATIONS")
    print("-" * 40)

    preds = predictor.predict_with_substitution_class(tokens, parser, top_k=max_predictions)
    if preds:
        for i, p in enumerate(preds, 1):
            print(f'  {i:2d}. "{p.unit_text}" ({p.score:.3f})')
    else:
        print("  (no predictions)")

    print()


def collect_spans(node, spans=None):
    """Recursively collect all span texts from parse tree."""
    if spans is None:
        spans = []

    spans.append(node.span.text)

    if node.left:
        collect_spans(node.left, spans)
    if node.right:
        collect_spans(node.right, spans)

    return spans


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phrase.py \"phrase to analyze\"")
        print("\nExamples:")
        print('  python analyze_phrase.py "we should go"')
        print('  python analyze_phrase.py "i want to"')
        print('  python analyze_phrase.py "she said that"')
        sys.exit(1)

    phrase = sys.argv[1]

    # Load models
    print("Loading models...")
    catalog = UnitCatalog()
    catalog.load('unit_catalog.pkl')

    predictor = SubstitutionPredictor(catalog, similarity_method='lin')
    parser = SimpleBidirParser(catalog, debug=False)

    # Analyze
    analyze(phrase, predictor, parser)


if __name__ == "__main__":
    main()
