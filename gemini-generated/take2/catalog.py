import re
import math
from collections import defaultdict, Counter

class SimpleTokenizer:
    def tokenize(self, text):
        # Simple whitespace and punctuation splitting
        # explicit keep of common punctuation as tokens
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        return text.lower().split()

class AggregatedCatalog:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        
        # PRIMARY STORAGE 
        # Unit -> { 'left': Counter(), 'right': Counter() }
        # We separate left/right distributions to preserve structural info [cite: 236]
        self.unit_distributions = defaultdict(lambda: {'left': Counter(), 'right': Counter()})
        
        # Global frequency for baseline probability
        self.unit_counts = Counter()
        self.total_tokens = 0

    def ingest(self, text):
        tokens = self.tokenizer.tokenize(text)
        self.total_tokens += len(tokens)
        
        # Sliding window to capture units of length 1 to 4 [cite: 163]
        MAX_N = 4
        
        for i in range(len(tokens)):
            for n in range(1, MAX_N + 1):
                if i + n > len(tokens): break
                
                # The Unit
                span = tuple(tokens[i : i+n])
                self.unit_counts[span] += 1
                
                # The Context (Immediate neighbors) [cite: 10]
                # We capture the immediate words to build the distribution
                left_word = tokens[i-1] if i > 0 else "<START>"
                right_word = tokens[i+n] if i+n < len(tokens) else "<END>"
                
                # Update Aggregated Distributions [cite: 239]
                self.unit_distributions[span]['left'][left_word] += 1
                self.unit_distributions[span]['right'][right_word] += 1

    def get_context_signature(self, span_tuple):
        """Returns the left/right distributions for a unit."""
        return self.unit_distributions.get(span_tuple, {'left': Counter(), 'right': Counter()})

    def get_all_units(self):
        return list(self.unit_distributions.keys())
