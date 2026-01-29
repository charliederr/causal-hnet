import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Import our modules (assuming you saved them as files, or paste them in)
# from data_loader import CorpusProcessor
# from expansion_engine import run_expansion_pass, compute_energy

# For this standalone script to work, I will include the minimal necessary
# wrapper classes and the scoring logic inline.

class JAXParser:
    def __init__(self, corpus_data):
        """
        Initialize with the artifacts from CorpusProcessor.
        """
        self.centroids = jnp.array(corpus_data['centroids'])
        self.adj_matrix = jnp.array(corpus_data['adj_matrix'])
        
        # Maps: "my money" -> ID
        self.unit_to_id = {v: k for k, v in corpus_data['unit_labels'].items()}
        self.id_to_unit = corpus_data['unit_labels']
        
        # Load BERT for encoding *new* sentences at runtime
        # (In production, share this instance with data_loader)
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.model.eval()

    def encode_context(self, left_text, right_text):
        """
        Encodes a new context "left [MASK] right" into a vector
        compatible with our JAX centroids.
        """
        text = f"{left_text} [MASK] {right_text}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return numpy array for JAX
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()

    def get_unit_score(self, span_text, left_context, right_context):
        """
        Scores a specific span as a unit using the Bidirectional Expansion logic.
        """
        # 1. Encode the specific context of this occurrence
        context_vec = self.encode_context(left_context, right_context)
        
        # 2. Run JAX Expansion
        # Note: We need to convert context_vec to JAX array
        ctx_vec_jax = jnp.array(context_vec)
        
        # Import or define the engine function here for access
        # (Assuming it's defined as in the previous step)
        # For simplicity, I'll inline a synchronous call to the logic
        
        # A. Expand Contexts
        # cosine sim
        c_norm = self.centroids / jnp.linalg.norm(self.centroids, axis=1, keepdims=True)
        v_norm = ctx_vec_jax / jnp.linalg.norm(ctx_vec_jax)
        sims = jnp.dot(c_norm, v_norm)
        ctx_mask = jax.nn.softmax(sims / 0.1) # Temperature
        
        # B. Expand Units
        unit_acts = jnp.dot(self.adj_matrix, ctx_mask)
        
        # C. Calculate Energy
        # E = -log(volume_units)
        # We also add a bonus if the span_text ITSELF is in the expansion.
        # The paper mentions: E = -log(freq) + context_mismatch
        
        volume_energy = -jnp.log(jnp.sum(unit_acts) + 1e-9)
        
        # Specific Unit Check:
        # Does the specific span "my money" actually fit this context cluster?
        # If the span isn't in our catalog, it gets a high energy penalty.
        if span_text in self.unit_to_id:
            unit_id = self.unit_to_id[span_text]
            # How much did the expansion "predict" this specific unit?
            specific_activation = unit_acts[unit_id]
            # Invert for energy (low activation = high energy)
            specific_energy = -jnp.log(specific_activation + 1e-9)
            
            # Combine: Total volume (is it a syntactic slot?) + Specific fit (does this word fit?)
            total_energy = 0.7 * specific_energy + 0.3 * volume_energy
        else:
            # Unknown unit -> High Energy
            total_energy = 100.0
            
        return float(total_energy)

    def parse(self, tokens):
        """
        Recursive top-down parser.
        Input: List of tokens e.g., ["I", "lost", "my", "money"]
        """
        
        # Memoization cache could go here
        
        def recursive_step(start, end):
            span_text = " ".join(tokens[start:end])
            
            # Base case: Single token is always a unit (leaf node)
            if end - start == 1:
                return {"type": "leaf", "text": span_text, "energy": 0.0}

            # 1. Test as Whole Unit
            # Get context for this span
            left_ctx = " ".join(tokens[:start])
            right_ctx = " ".join(tokens[end:])
            
            unit_energy = self.get_unit_score(span_text, left_ctx, right_ctx)
            
            # 2. Find Best Split
            best_split_energy = float('inf')
            best_split_node = None
            
            # Try all split points
            for i in range(start + 1, end):
                left_node = recursive_step(start, i)
                right_node = recursive_step(i, end)
                
                # Simple sum of energies for split (Constituency assumption)
                # You can add a penalty for splitting to encourage grouping
                split_energy = left_node['energy'] + right_node['energy']
                
                if split_energy < best_split_energy:
                    best_split_energy = split_energy
                    best_split_node = {"type": "split", "left": left_node, "right": right_node, "energy": split_energy}

            # 3. Decision
            # If the unit energy is lower (better) than splitting, keep it whole.
            # We usually need a bias/threshold because splitting always reduces energy in simple sums.
            # Let's add a "Composition Penalty" for splitting.
            SPLIT_PENALTY = 2.0 
            
            if unit_energy < (best_split_energy + SPLIT_PENALTY):
                return {"type": "unit", "text": span_text, "energy": unit_energy, "children": best_split_node} 
                # Note: We keep children just for visualization, but structurally it's a Unit.
            else:
                return best_split_node

        return recursive_step(0, len(tokens))

    def print_tree(self, node, depth=0):
        indent = "  " * depth
        if node['type'] == 'leaf':
            print(f"{indent}[{node['text']}]")
        elif node['type'] == 'unit':
            print(f"{indent}(UNIT: {node['text']} | E={node['energy']:.2f})")
            # If you want to see internal structure of units, uncomment:
            # self.print_tree(node['children'], depth + 1)
        else:
            print(f"{indent}(SPLIT E={node['energy']:.2f})")
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Load Data (Simulated or from previous step)
    # Ideally: from data_loader import CorpusProcessor; p = CorpusProcessor(); ...
    # Here we mock the output of processor.export() for a runnable demo
    
    print("Initializing Parser...")
    
    # Mock Data matching the previous script's logic
    # "my money" (ID 0) fits in Cluster 0
    # "lost my" (ID 1) fits in Cluster 1 (we make it weaker)
    mock_centroids = np.random.normal(size=(5, 64)) # 5 clusters
    mock_adj = np.zeros((3, 5))
    mock_adj[0, 0] = 10.0 # "my money" strong in cluster 0
    mock_adj[1, 1] = 2.0  # "lost my" weak in cluster 1
    
    data_pack = {
        'centroids': mock_centroids,
        'adj_matrix': mock_adj,
        'unit_labels': {0: "my money", 1: "lost my", 2: "money"}
    }
    
    parser = JAXParser(data_pack)
    
    # 2. Test Sentence
    sentence = ["I", "lost", "my", "money"]
    print(f"\nParsing: {sentence}")
    
    # 3. Parse
    result = parser.parse(sentence)
    
    # 4. Display
    print("\n--- Parse Result ---")
    parser.print_tree(result)
