#!/usr/bin/env python3
import pickle
from collections import Counter
from dataclasses import dataclass

@dataclass
class ContextPattern:
    left_words: Counter
    right_words: Counter
    count: int

with open('unit_catalog.pkl', 'rb') as f:
    units = pickle.load(f)

test_units = ['go out', 'my money', 'did you', 'go', 'out', 'my', 'money', 'lost my']

for u in test_units:
    if u in units:
        pattern = units[u]
        left_top = dict(pattern.left_words.most_common(3))
        right_top = dict(pattern.right_words.most_common(3))
        print(f"{u!r}: count={pattern.count}")
        print(f"  left: {left_top}")
        print(f"  right: {right_top}")
    else:
        print(f"{u!r}: NOT IN CATALOG")
