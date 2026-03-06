#!/usr/bin/env python3
"""
Test batch parsing only (without predictor) to isolate memory issues.
"""

import time
import sys

# Import ContextPattern FIRST to fix pickle loading
from bidir_simple import UnitCatalog, SimpleBidirParser, ContextPattern

print("=" * 60)
print("BATCH PARSE TEST (without predictor)")
print("=" * 60)

# Load catalog
print("\nLoading unit catalog...")
catalog = UnitCatalog()
try:
    catalog.load('unit_catalog.pkl')
except FileNotFoundError:
    print("Error: unit_catalog.pkl not found. Run bidir_simple.py first.")
    sys.exit(1)

# Build GPU index for catalog
print("\nBuilding GPU index for catalog...")
start = time.time()
catalog.build_gpu_index()
gpu_index_time = time.time() - start
print(f"GPU index time: {gpu_index_time:.2f}s")

# Clear cache after building index
import torch
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    print("Cleared MPS cache")

# Create parser
parser = SimpleBidirParser(catalog, debug=False)

# Test prompts
test_prompts = [
    ["i", "want", "to"],
    ["she", "said", "that"],
]

print("\n" + "=" * 60)
print("TESTING BATCH vs SEQUENTIAL PARSING")
print("=" * 60)

for prompt in test_prompts:
    prompt_text = ' '.join(prompt)
    print(f"\nPROMPT: '{prompt_text}'")

    # Clear cache before each test
    torch.mps.empty_cache()

    # Test BATCH parsing (new fast method)
    print("  Running batch parse...")
    t2 = time.time()
    try:
        tree_batch = parser.parse_batch(prompt)
        t3 = time.time()
        print(f"  Batch parse time: {t3-t2:.3f}s")
    except Exception as e:
        t3 = time.time()
        print(f"  Batch parse FAILED after {t3-t2:.3f}s: {e}")
        tree_batch = None

    # Clear cache between methods
    torch.mps.empty_cache()

    # Test OLD sequential parsing for comparison
    print("  Running sequential parse...")
    t4 = time.time()
    tree_seq = parser.parse(prompt)
    t5 = time.time()
    print(f"  Sequential parse time: {t5-t4:.3f}s")

    if tree_batch:
        print(f"  Speedup: {(t5-t4)/(t3-t2+0.001):.1f}x")

print("\nDone!")
