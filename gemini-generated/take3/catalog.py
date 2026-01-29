from collections import defaultdict, Counter
from tokenizer import SimpleTokenizer

class AggregatedCatalog:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        
        # PRIMARY STORAGE
        # 1. Unit -> Frequency
        self.unit_counts = Counter()
        
        # 2. Inverted Index: (left_word, right_word) -> Set of Units
        # This allows us to instantly answer: "What else fits here?"
        self.context_to_units = defaultdict(set)
        
        # 3. Unit -> Contexts (for reverse lookup/entropy calc)
        self.unit_to_contexts = defaultdict(list)

    def ingest(self, text):
        tokens = self.tokenizer.tokenize(text)
        total_tokens = len(tokens)
        print(f"Ingesting {total_tokens} tokens...")
        
        # Sliding window for units up to length 4
        MAX_N = 4
        
        for i in range(len(tokens)):
            for n in range(1, MAX_N + 1):
                if i + n > len(tokens): break
                
                # Define Unit and Context
                unit_tuple = tuple(tokens[i : i+n])
                
                # Context words (use <S> for boundaries)
                left_w = tokens[i-1] if i > 0 else "<START>"
                right_w = tokens[i+n] if i+n < len(tokens) else "<END>"
                context_tuple = (left_w, right_w)
                
                # Update Stats
                self.unit_counts[unit_tuple] += 1
                self.context_to_units[context_tuple].add(unit_tuple)
                self.unit_to_contexts[unit_tuple].append(context_tuple)

    def get_substitutes(self, left_ctx, right_ctx):
        """Returns all units that have appeared in this exact context."""
        return self.context_to_units.get((left_ctx, right_ctx), set())

    def get_unit_freq(self, unit_tuple):
        return self.unit_counts[unit_tuple]
