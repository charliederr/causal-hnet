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

for u in ['on', 'friday', 'on friday', 'out on friday', 'go out on', 'go out on friday']:
    if u in units:
        print(f"{u!r}: count={units[u].count}")
    else:
        print(f"{u!r}: NOT IN CATALOG")
