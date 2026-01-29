import math

class ExpansionEngine:
    def __init__(self, catalog):
        self.catalog = catalog

    def calculate_energy(self, target_unit, left_ctx, right_ctx):
        """
        Calculates Energy E = -log(Expansion Volume)
        
        Expansion Volume = Sum of frequencies of ALL units that fit 
        in the current (left, right) context.
        """
        # 1. Find Substitutes (The "Expansion Set")
        # "What other units fit in (left_ctx, right_ctx)?"
        substitutes = self.catalog.get_substitutes(left_ctx, right_ctx)
        
        # 2. Calculate Volume
        # We sum the frequencies of the substitutes.
        # Common slots (e.g., "the ___ of") will have massive volume.
        volume = 0
        for unit in substitutes:
            volume += self.catalog.get_unit_freq(unit)
            
        # 3. Handle Novelty
        # If the exact context was never seen, volume is 0.
        if volume == 0:
            # Fallback: Is the target unit itself common?
            # If so, maybe this is just a rare context for a common unit.
            unit_freq = self.catalog.get_unit_freq(target_unit)
            if unit_freq > 0:
                # High energy (bad), but not infinite
                return 15.0 - math.log(unit_freq)
            else:
                # Total unknown -> Max Energy
                return 20.0

        # 4. Energy Score
        # Higher volume = Lower Energy (Better Unit)
        return -math.log(volume)
