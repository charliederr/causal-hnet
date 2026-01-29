class TopDownParser:
    def __init__(self, engine):
        self.engine = engine

    def parse(self, tokens):
        """
        Recursive top-down parsing.
        """
        # Memoization dict
        memo = {}

        def recursive_solve(start, end):
            # Check memo
            if (start, end) in memo: return memo[(start, end)]

            # Base case
            if end - start == 1:
                return {'type': 'leaf', 'text': tokens[start], 'energy': 0.0}

            span = tuple(tokens[start:end])
            
            # 1. Get Context for this instance
            left_ctx = tokens[start-1] if start > 0 else "<START>"
            right_ctx = tokens[end] if end < len(tokens) else "<END>"

            # 2. Test as Unit (The Expansion Score) [cite: 180]
            unit_energy = self.engine.bidirectional_expansion(span, left_ctx, right_ctx)
            
            # 3. Test Splits [cite: 185]
            best_split_energy = float('inf')
            best_split_node = None
            
            for i in range(start+1, end):
                left_res = recursive_solve(start, i)
                right_res = recursive_solve(i, end)
                
                # Energy is additive for independent parts
                split_energy = left_res['energy'] + right_res['energy']
                
                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split_node = {'type': 'split', 'left': left_res, 'right': right_res, 'energy': split_energy}

            # 4. Compare [cite: 191]
            # "Structure Bonus": We prefer keeping things together if energies are close.
            # This constant represents the "cost" of breaking a dependency.
            STRUCTURE_BIAS = 2.0
            
            if unit_energy < (best_split_energy + STRUCTURE_BIAS):
                res = {'type': 'unit', 'text': " ".join(span), 'energy': unit_energy, 'children': best_split_node}
            else:
                res = best_split_node

            memo[(start, end)] = res
            return res

        return recursive_solve(0, len(tokens))

    def print_tree(self, node, indent=0):
        sp = "  " * indent
        if node['type'] == 'leaf':
            print(f"{sp}[{node['text']}]")
        elif node['type'] == 'unit':
            print(f"{sp}(UNIT '{node['text']}' E={node['energy']:.2f})")
            # You can recursively print children here if you want to see the substructure
        else:
            print(f"{sp}(SPLIT E={node['energy']:.2f})")
            self.print_tree(node['left'], indent+1)
            self.print_tree(node['right'], indent+1)
