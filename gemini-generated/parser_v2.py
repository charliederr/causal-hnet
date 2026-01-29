import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

class JAXParser:
    def __init__(self, corpus_data):
        self.centroids = jnp.array(corpus_data['centroids'])
        self.adj_matrix = jnp.array(corpus_data['adj_matrix'])
        self.unit_to_id = {v: k for k, v in corpus_data['unit_labels'].items()}
        self.id_to_unit = corpus_data['unit_labels']
        
        # Load BERT
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.model.eval()

    def encode_context(self, left_text, right_text):
        text = f"{left_text} [MASK] {right_text}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()

    def get_unit_score(self, span_text, left_context, right_context):
        # 1. Encode Context
        context_vec = self.encode_context(left_context, right_context)
        ctx_vec_jax = jnp.array(context_vec)
        
        # 2. Expand Contexts (Cosine Similarity)
        # Normalize vectors for similarity search
        c_norm = self.centroids / (jnp.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9)
        v_norm = ctx_vec_jax / (jnp.linalg.norm(ctx_vec_jax) + 1e-9)
        
        sims = jnp.dot(c_norm, v_norm)
        
        # Softmax to get probability distribution over clusters
        ctx_mask = jax.nn.softmax(sims / 0.1) 
        
        # 3. Expand Units (Project context mask into unit space)
        # This tells us: "Given this context, how many known units fit here?"
        unit_acts = jnp.dot(self.adj_matrix, ctx_mask)
        
        # 4. Compute Volume Energy (The "Slot Score")
        # High sum(unit_acts) means this is a very common syntactic slot (e.g., a noun phrase position).
        # We want Low Energy for High Volume.
        volume_energy = -jnp.log(jnp.sum(unit_acts) + 1e-9)
        
        # 5. Compute Specific Fit
        if span_text in self.unit_to_id:
            # KNOWN UNIT: We combine the general slot score with the specific unit's activation
            unit_id = self.unit_to_id[span_text]
            specific_activation = unit_acts[unit_id]
            specific_energy = -jnp.log(specific_activation + 1e-9)
            
            # Weighted average: Specific fit is more important
            total_energy = 0.6 * specific_energy + 0.4 * volume_energy
        else:
            # UNKNOWN UNIT: This is the fix. 
            # Previously, we returned 20.0 here, which killed discovery.
            # Now, we rely on volume_energy (is this a valid slot?) + a small novelty penalty.
            
            # If volume_energy is very low (e.g. -5.0), it means "Lots of things go here".
            # So if we see a new phrase here, it's likely a valid unit filling that slot.
            NOVELTY_PENALTY = 3.5 
            total_energy = volume_energy + NOVELTY_PENALTY
            
        return float(total_energy)

    def parse(self, tokens):
        # Memoization cache
        memo = {}

        def recursive_step(start, end):
            # Check cache
            state_key = (start, end)
            if state_key in memo: return memo[state_key]

            span_text = " ".join(tokens[start:end])
            
            # Base case: Single token
            if end - start == 1:
                return {"type": "leaf", "text": span_text, "energy": 0.0}

            # 1. Test as Unit
            left_ctx = " ".join(tokens[:start])
            right_ctx = " ".join(tokens[end:])
            unit_energy = self.get_unit_score(span_text, left_ctx, right_ctx)
            
            # 2. Find Best Split
            best_split_energy = float('inf')
            best_split_node = None
            
            for i in range(start + 1, end):
                left_node = recursive_step(start, i)
                right_node = recursive_step(i, end)
                
                # Standard split summation
                split_energy = left_node['energy'] + right_node['energy']
                
                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split_node = {
                        "type": "split", 
                        "left": left_node, 
                        "right": right_node, 
                        "energy": split_energy
                    }

            # 3. Decision
            # SPLIT_PENALTY: The cost of breaking a bond.
            # Increased to 5.0 to encourage larger units.
            SPLIT_PENALTY = 5.0 
            
            # Note: We prefer units if their energy is competitive
            if unit_energy < (best_split_energy + SPLIT_PENALTY):
                result = {"type": "unit", "text": span_text, "energy": unit_energy, "children": best_split_node}
            else:
                result = best_split_node
                
            memo[state_key] = result
            return result

        return recursive_step(0, len(tokens))

    def print_tree(self, node, depth=0):
        indent = "  " * depth
        if node['type'] == 'leaf':
            print(f"{indent}[{node['text']}]")
        elif node['type'] == 'unit':
            print(f"{indent}(UNIT: {node['text']} | E={node['energy']:.2f})")
        else:
            print(f"{indent}(SPLIT E={node['energy']:.2f})")
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
