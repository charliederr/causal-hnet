import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import warnings

# Suppress harmless warnings about BERT weights
warnings.filterwarnings("ignore", category=UserWarning)

class JAXParser:
    def __init__(self, corpus_data):
        """
        Initialize with the artifacts from CorpusProcessor.
        """
        # Ensure centroids are JAX arrays
        self.centroids = jnp.array(corpus_data['centroids'])
        self.adj_matrix = jnp.array(corpus_data['adj_matrix'])
        
        # Maps: "my money" -> ID
        self.unit_to_id = {v: k for k, v in corpus_data['unit_labels'].items()}
        self.id_to_unit = corpus_data['unit_labels']
        
        # Load BERT for encoding *new* sentences at runtime
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
        # Return numpy array for JAX, flattened to 1D
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()

    def get_unit_score(self, span_text, left_context, right_context):
        """
        Scores a specific span as a unit using the Bidirectional Expansion logic.
        """
        # 1. Encode the specific context of this occurrence
        context_vec = self.encode_context(left_context, right_context)
        
        # 2. Run JAX Expansion
        ctx_vec_jax = jnp.array(context_vec)
        
        # A. Expand Contexts
        # Cosine similarity: dot product of normalized vectors
        # (K, D)
        c_norm = self.centroids / (jnp.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-9)
        # (D,)
        v_norm = ctx_vec_jax / (jnp.linalg.norm(ctx_vec_jax) + 1e-9)
        
        # Dot product: (K, D) @ (D,) -> (K,)
        sims = jnp.dot(c_norm, v_norm)
        
        # Softmax with temperature to create a probability mask
        ctx_mask = jax.nn.softmax(sims / 0.1) 
        
        # B. Expand Units
        # (N_units, K) @ (K,) -> (N_units,)
        unit_acts = jnp.dot(self.adj_matrix, ctx_mask)
        
        # C. Calculate Energy
        # E = -log(volume_units)
        volume_energy = -jnp.log(jnp.sum(unit_acts) + 1e-9)
        
        # Specific Unit Check:
        # Does the specific span (e.g., "my money") actually fit this context cluster?
        if span_text in self.unit_to_id:
            unit_id = self.unit_to_id[span_text]
            # How much did the expansion "predict" this specific unit?
            specific_activation = unit_acts[unit_id]
            # Invert for energy (low activation = high energy)
            specific_energy = -jnp.log(specific_activation + 1e-9)
            
            # Combine: Total volume (is it a syntactic slot?) + Specific fit (does this word fit?)
            total_energy = 0.7 * specific_energy + 0.3 * volume_energy
        else:
            # Unknown unit -> High Energy penalty
            total_energy = 20.0 # Cap high energy to avoid Infs ruining math
            
        return float(total_energy)

    def parse(self, tokens):
        """
        Recursive top-down parser.
        Input: List of tokens e.g., ["I", "lost", "my", "money"]
        """
        # Memoization cache could go here for efficiency
        
        def recursive_step(start, end):
            span_text = " ".join(tokens[start:end])
            
            # Base case: Single token is always a unit (leaf node)
            if end - start == 1:
                return {"type": "leaf", "text": span_text, "energy": 0.0}

            # 1. Test as Whole Unit
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
            # If unit_energy is competitive with splitting, keep it.
            # SPLIT_PENALTY encourages larger units.
            SPLIT_PENALTY = 3.0 
            
            if unit_energy < (best_split_energy + SPLIT_PENALTY):
                return {
                    "type": "unit", 
                    "text": span_text, 
                    "energy": unit_energy, 
                    "children": best_split_node
                } 
            else:
                return best_split_node

        return recursive_step(0, len(tokens))

    def print_tree(self, node, depth=0):
        indent = "  " * depth
        if node['type'] == 'leaf':
            print(f"{indent}[{node['text']}]")
        elif node['type'] == 'unit':
            print(f"{indent}(UNIT: {node['text']} | E={node['energy']:.2f})")
            # Recursively print children to see internal structure if desired
            # if node['children']: self.print_tree(node['children'], depth + 1)
        else:
            print(f"{indent}(SPLIT E={node['energy']:.2f})")
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Initializing Parser...")
    
    # --- MOCK DATA SETUP ---
    # BERT-tiny uses dimension 128. We MUST match this.
    EMBEDDING_DIM = 128 
    NUM_CLUSTERS = 5
    
    # Random centroids to simulate context clusters
    mock_centroids = np.random.normal(size=(NUM_CLUSTERS, EMBEDDING_DIM))
    
    # Adjacency Matrix: [Units, Clusters]
    # Unit 0 ("my money") fits Cluster 0
    # Unit 1 ("lost my") fits Cluster 1
    # Unit 2 ("money") fits Cluster 2
    mock_adj = np.zeros((3, NUM_CLUSTERS))
    mock_adj[0, 0] = 10.0 
    mock_adj[1, 1] = 2.0  
    mock_adj[2, 2] = 5.0
    
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
