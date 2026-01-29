import math

class ExpansionEngine:
    def __init__(self, catalog):
        self.catalog = catalog
        self.units = catalog.get_all_units()

    def jaccard_similarity(self, counter_a, counter_b, top_k=20):
        """
        Computes similarity between two context distributions.
        Implementation of Challenge 2 solution.
        """
        # Optimization: Only compare top-k most frequent context words
        # to avoid noise from rare accidental co-occurrences.
        keys_a = set([k for k, v in counter_a.most_common(top_k)])
        keys_b = set([k for k, v in counter_b.most_common(top_k)])
        
        if not keys_a or not keys_b:
            return 0.0
            
        intersection = len(keys_a.intersection(keys_b))
        union = len(keys_a.union(keys_b))
        
        return intersection / union

    def get_unit_similarity(self, unit_a, unit_b):
        """
        How interchangeable are two units? 
        Measured by the similarity of their context distributions.
        """
        sig_a = self.catalog.get_context_signature(unit_a)
        sig_b = self.catalog.get_context_signature(unit_b)
        
        # Score is average of left-context match and right-context match
        sim_left = self.jaccard_similarity(sig_a['left'], sig_b['left'])
        sim_right = self.jaccard_similarity(sig_b['right'], sig_b['right'])
        
        return (sim_left + sim_right) / 2.0

    def bidirectional_expansion(self, target_unit, current_left, current_right):
        """
        The Core Algorithm [cite: 58-79].
        
        1. Start with the target unit.
        2. Find 'Peers': Other units that share the same context profile (Paradigmatic relations).
        3. Score based on the size/strength of this set.
        """
        # 1. Context Expansion (Implicit in this approach)
        # We construct a 'query context' from the current sentence
        query_left = current_left
        query_right = current_right
        
        # 2. Unit Expansion [cite: 73]
        # We scan the catalog for units that fit this context profile.
        # Since we are "from scratch" without vector indexing, this is an O(N) scan.
        # (For production, this would need an Inverted Index [cite: 204])
        
        expanded_units = []
        
        # Heuristic: We only scan units that have actually appeared 
        # with at least one of our context words to speed this up.
        
        # STRICT MATCHING (Prototype V1 style [cite: 162])
        # Does the candidate unit strictly allow these context words?
        for candidate in self.units:
            sig = self.catalog.get_context_signature(candidate)
            
            # Check if candidate allows the current left/right words
            # (Relaxed check: simply must have seen them at least once)
            has_left = (query_left in sig['left']) or (query_left == "<START>")
            has_right = (query_right in sig['right']) or (query_right == "<END>")
            
            if has_left and has_right:
                # 3. Soft Expansion: Weight by how "typical" this context is for the unit
                weight_left = sig['left'][query_left]
                weight_right = sig['right'][query_right]
                score = math.log(weight_left + 1) + math.log(weight_right + 1)
                expanded_units.append(score)

        # 4. Calculate Energy [cite: 80]
        # E = -log(unit_expansion_size)
        # We use sum of scores as "size"
        expansion_volume = sum(expanded_units)
        
        # Avoid log(0)
        energy = -math.log(expansion_volume + 0.0001)
        
        return energy
