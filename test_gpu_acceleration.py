#!/usr/bin/env python3
"""
Test GPU acceleration for substitution class computation.
Compares MPS (Apple Metal) performance against CPU baseline.
"""

import time
import sys

# Import ContextPattern FIRST to fix pickle loading
from bidir_simple import UnitCatalog, SimpleBidirParser, ContextPattern

# Import predictor (this will print GPU detection status)
print("=" * 60)
print("GPU ACCELERATION TEST")
print("=" * 60)
print()

from substitution_predictor import SubstitutionPredictor, HAS_TORCH, DEVICE

# Load catalog
print("\nLoading unit catalog...")
catalog = UnitCatalog()
try:
    catalog.load('unit_catalog.pkl')
except FileNotFoundError:
    print("Error: unit_catalog.pkl not found. Run bidir_simple.py first.")
    sys.exit(1)

# Build GPU index for catalog (used by parser)
print("\nBuilding GPU index for catalog...")
start = time.time()
catalog.build_gpu_index()
gpu_index_time = time.time() - start
print(f"GPU index time: {gpu_index_time:.2f}s")

# Create predictor
print("\nCreating predictor (building GPU tensors)...")
start = time.time()
predictor = SubstitutionPredictor(catalog)
init_time = time.time() - start
print(f"Predictor initialization time: {init_time:.2f}s")

# Create parser
parser = SimpleBidirParser(catalog, debug=False)

# Test prompts
test_prompts = [
    ["i", "want", "to"],
    ["she", "said", "that"],
    ["we", "should", "go"],
]

print("\n" + "=" * 60)
print("TESTING SUBSTITUTION CLASS PREDICTION")
print("=" * 60)

for prompt in test_prompts:
    prompt_text = ' '.join(prompt)
    print(f"\nPROMPT: '{prompt_text}'")

    # Clear cache to get accurate timing
    if hasattr(predictor, '_subclass_cache'):
        predictor._subclass_cache.clear()

    # Test DIRECT substitution class computation (no parsing)
    t0 = time.time()
    sub_class = predictor.find_substitution_class(prompt_text, max_members=30)
    t1 = time.time()

    print(f"  Direct substitution class time: {t1-t0:.3f}s (found {len(sub_class)} members)")
    if sub_class:
        print(f"  Top members: {[m[0] for m in sub_class[:5]]}")

    # Test BATCH parsing (new fast method)
    t2 = time.time()
    tree_batch = parser.parse_batch(prompt)
    t3 = time.time()
    print(f"  Batch parse time: {t3-t2:.3f}s")

    # Test OLD sequential parsing for comparison
    t4 = time.time()
    tree_seq = parser.parse(prompt)
    t5 = time.time()
    print(f"  Sequential parse time: {t5-t4:.3f}s")
    print(f"  Speedup: {(t5-t4)/(t3-t2+0.001):.1f}x")

# Compare with cached (should be near-instant)
print("\n" + "=" * 60)
print("TESTING WITH CACHED SUBSTITUTION CLASSES")
print("=" * 60)

for prompt in test_prompts:
    prompt_text = ' '.join(prompt)

    start = time.time()
    preds = predictor.predict_with_substitution_class(prompt, parser, top_k=5)
    elapsed = time.time() - start

    print(f"'{prompt_text}': {elapsed:.3f}s (cached)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"PyTorch available: {HAS_TORCH}")
print(f"Device: {DEVICE}")
print(f"GPU tensors built: {getattr(predictor, 'gpu_available', False)}")
