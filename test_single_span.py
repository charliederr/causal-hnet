#!/usr/bin/env python3
"""Test parsing a single span in isolation."""

import sys
sys.path.insert(0, '.')

# Import everything from the module so pickle can find ContextPattern
from hamiltonian_parser import *

catalog = UnitCatalog()
catalog.load('unit_catalog.pkl')

parser = HamiltonianParser(catalog, debug=True, max_iters=0, use_expansion=False)

# Test just "go out on friday" in isolation
test_text = "go out on friday"
tokens = test_text.split()

print(f"Testing: \"{test_text}\"")
print(f"Tokens: {tokens}")
print(f"Length: {len(tokens)}")
print()

result = parser.parse(tokens)

print("\nResult:")
print(f"  is_unit: {result.is_unit}")
print(f"  unit_energy: {result.unit_energy:.2f}")
if not result.is_unit:
    print(f"  split_energy: {result.split_energy:.2f}")
    print(f"  left: \"{result.left_child.span.text}\"")
    print(f"  right: \"{result.right_child.span.text}\"")
