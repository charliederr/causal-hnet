import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict, Counter
from tqdm import tqdm

# Configuration
MODEL_NAME = "prajjwal1/bert-tiny"  # Tiny model for fast CPU prototyping
MAX_NGRAM = 4                       # Max length of units (words)
CONTEXT_WINDOW = 3                  # Tokens to left/right
NUM_CLUSTERS = 64                   # Small K for prototype (increase for real data)
MIN_FREQ = 2                        # Minimum occurrences to keep a unit

class CorpusProcessor:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        
        # Stores: unit_string -> list of context_vectors
        self.unit_contexts_map = defaultdict(list)
        # Stores: unit_string -> ID
        self.unit_to_id = {}
        self.id_to_unit = {}
        
        # JAX-ready artifacts
        self.centroids = None
        self.adjacency_matrix = None

    def _get_embeddings(self, text_batch):
        """Generates embeddings for a batch of context strings."""
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token as sentence/context representation
        return outputs.last_hidden_state[:, 0, :].numpy()

    def ingest_text(self, text_corpus):
        """
        Parses text to extract (Unit, Context) pairs.
        Context format: "left_context [MASK] right_context"
        """
        print("Tokenizing corpus...")
        tokens = self.tokenizer.tokenize(text_corpus)
        
        # Sliding window extraction
        print(f"Extracting n-grams (1-{MAX_NGRAM})...")
        
        # Temporary buffers for batch processing
        context_strings = []
        unit_references = [] # (unit_str)

        for i in range(len(tokens)):
            # Try different n-gram lengths at this position
            for n in range(1, MAX_NGRAM + 1):
                if i + n > len(tokens): break
                
                # The candidate unit
                unit_tokens = tokens[i : i+n]
                unit_str = self.tokenizer.convert_tokens_to_string(unit_tokens).strip()
                
                # The Context
                # Left: [i-window : i]
                start = max(0, i - CONTEXT_WINDOW)
                left_tokens = tokens[start : i]
                left_str = self.tokenizer.convert_tokens_to_string(left_tokens)
                
                # Right: [i+n : i+n+window]
                end = min(len(tokens), i + n + CONTEXT_WINDOW)
                right_tokens = tokens[i+n : end]
                right_str = self.tokenizer.convert_tokens_to_string(right_tokens)
                
                # Create a "Context String" for BERT
                # e.g. "I lost [MASK] yesterday"
                ctx_str = f"{left_str} [MASK] {right_str}"
                
                context_strings.append(ctx_str)
                unit_references.append(unit_str)

        print(f"Vectorizing {len(context_strings)} contexts (this may take a moment)...")
        
        # Process in batches to avoid OOM
        BATCH_SIZE = 32
        for i in tqdm(range(0, len(context_strings), BATCH_SIZE)):
            batch_ctx = context_strings[i : i+BATCH_SIZE]
            batch_units = unit_references[i : i+BATCH_SIZE]
            
            embeddings = self._get_embeddings(batch_ctx)
            
            for unit, vec in zip(batch_units, embeddings):
                self.unit_contexts_map[unit].append(vec)

        # Filter low frequency units
        initial_count = len(self.unit_contexts_map)
        self.unit_contexts_map = {k: v for k, v in self.unit_contexts_map.items() if len(v) >= MIN_FREQ}
        print(f"Filtered units: {initial_count} -> {len(self.unit_contexts_map)}")

    def build_jax_structures(self):
        """
        1. Clusters all context vectors to find Centroids (Matrix C).
        2. Builds Unit-Context Adjacency (Matrix A).
        """
        print("Clustering contexts...")
        # Collect ALL context vectors from ALL units for global clustering
        all_vectors = []
        for vec_list in self.unit_contexts_map.values():
            all_vectors.extend(vec_list)
        
        all_vectors = np.array(all_vectors)
        
        # K-Means Clustering
        kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init="auto")
        kmeans.fit(all_vectors)
        
        self.centroids = kmeans.cluster_centers_ # This is Matrix C
        print(f"Generated Context Matrix C: shape {self.centroids.shape}")

        print("Building Adjacency Matrix...")
        # Index Units
        sorted_units = sorted(self.unit_contexts_map.keys())
        self.unit_to_id = {u: i for i, u in enumerate(sorted_units)}
        self.id_to_unit = {i: u for i, u in enumerate(sorted_units)}
        
        # Build Matrix A (Rows=Units, Cols=Clusters)
        num_units = len(sorted_units)
        self.adjacency_matrix = np.zeros((num_units, NUM_CLUSTERS), dtype=np.float32)
        
        for unit, vectors in self.unit_contexts_map.items():
            u_id = self.unit_to_id[unit]
            
            # Predict which cluster each instance of this unit belongs to
            vectors = np.array(vectors)
            cluster_ids = kmeans.predict(vectors)
            
            # Count frequency in each cluster
            counts = Counter(cluster_ids)
            for c_id, freq in counts.items():
                self.adjacency_matrix[u_id, c_id] = freq
                
        # Optional: Normalize rows or leave as raw counts?
        # Raw counts preserve "mass" (more frequent units -> stronger signal)
        print(f"Generated Adjacency Matrix A: shape {self.adjacency_matrix.shape}")

    def export(self):
        """Returns the artifacts needed for the JAX engine."""
        return {
            "centroids": self.centroids,         # (K, Dim)
            "adj_matrix": self.adjacency_matrix, # (N_units, K)
            "unit_labels": self.id_to_unit       # Dict mapping ID -> String
        }

# --- usage ---
if __name__ == "__main__":
    # Sample corpus (polysemy test)
    sample_text = """
    I lost my money yesterday. I lost my keys today.
    Did you go out with Sarah? We should go out tonight.
    I go out the door. He needs to go out the exit.
    Where is my money? I need my money now.
    """
    
    processor = CorpusProcessor()
    processor.ingest_text(sample_text)
    processor.build_jax_structures()
    
    data = processor.export()
    
    print("\n--- Sanity Check ---")
    print("Centroids Shape:", data['centroids'].shape)
    print("Adjacency Shape:", data['adj_matrix'].shape)
    print("Sample Units:", list(data['unit_labels'].values())[:5])
