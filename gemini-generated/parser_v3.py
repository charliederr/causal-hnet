class TopDownParser:
    def __init__(self, engine):
        self.engine = engine

    def parse(self, tokens):
        memo = {}

        def recursive_solve(start, end):
            state = (start, end)
            if state in memo: return memo[state]

            # Base case: Single token is a leaf
            if end - start == 1:
                return {'type': 'leaf', 'text': tokens[start], 'energy': 0.0}

            span_tuple = tuple(tokens[start:end])
            span_text = " ".join(span_tuple)
            
            # 1. Get Context
            left_ctx = tokens[start-1] if start > 0 else "<START>"
            right_ctx = tokens[end] if end < len(tokens) else "<END>"

            # 2. Test as Unit (Expansion Score)
            unit_energy = self.engine.calculate_energy(span_tuple, left_ctx, right_ctx)
            
            # 3. Test Splits
            best_split_energy = float('inf')
            best_split_node = None
            
            for i in range(start+1, end):
                left_res = recursive_solve(start, i)
                right_res = recursive_solve(i, end)
                
                split_energy = left_res['energy'] + right_res['energy']
                
                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split_node = {'type': 'split', 'left': left_res, 'right': right_res, 'energy': split_energy}

            # 4. Compare
            # Structure Bias: How much cheaper must a unit be to win over a split?
            # A lower bias (e.g. 1.0) encourages more splitting. 
            # A higher bias (e.g. 5.0) forces larger chunks.
            STRUCTURE_BIAS = 2.5
            
            if unit_energy < (best_split_energy + STRUCTURE_BIAS):
                res = {'type': 'unit', 'text': span_text, 'energy': unit_energy, 'children': best_split_node}
            else:
                res = best_split_node

            memo[state] = res
            return res

        return recursive_solve(0, len(tokens))

    def print_tree(self, node, indent=0):
        sp = "  " * indent
        if node['type'] == 'leaf':
            print(f"{sp}[{node['text']}]")
        elif node['type'] == 'unit':
            print(f"{sp}(UNIT '{node['text']}' E={node['energy']:.2f})")
        else:
            print(f"{sp}(SPLIT E={node['energy']:.2f})")
            self.print_tree(node['left'], indent+1)
            self.print_tree(node['right'], indent+1)
