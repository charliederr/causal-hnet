import os
import random
import jax.numpy as jnp
import numpy as np
from scipy.stats import entropy
import data_loader  # Import to tweak settings
from data_loader import CorpusProcessor
from parser_v2 import JAXParser

def analyze_clusters(adj_matrix, unit_labels, n_top_words=8, num_clusters_to_show=10):
    """
    Reverse engineers the 'meaning' of clusters by seeing what words live in them.
    adj_matrix: (Num_Units, Num_Clusters)
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS 1: CONTEXT CLUSTER SEMANTICS")
    print(f"What kind of 'slots' did the system discover?")
    print(f"{'='*60}")

    num_units, num_clusters = adj_matrix.shape
    
    # Sort clusters by "volume" (how much text they account for)
    cluster_volumes = np.sum(adj_matrix, axis=0)
    sorted_cluster_ids = np.argsort(cluster_volumes)[::-1] # Descending

    # Convert unit labels to list for indexing
    id_to_unit = unit_labels
    
    for i in range(min(num_clusters_to_show, num_clusters)):
        c_id = sorted_cluster_ids[i]
        vol = cluster_volumes[c_id]
        
        # Get the column for this cluster
        col_data = adj_matrix[:, c_id]
        
        # Find indices of units with highest counts in this cluster
        top_unit_indices = np.argsort(col_data)[::-1][:n_top_words]
        
        # Retrieve words
        top_words = [id_to_unit[idx] for idx in top_unit_indices if col_data[idx] > 0]
        
        if not top_words: continue

        print(f"\n[Cluster #{c_id}] (Volume: {vol:.0f})")
        print(f"  Top Residents: {', '.join(top_words)}")
        print(f"  Interpretation: Likely a slot for '{top_words[0]}' types")

def analyze_unit_versatility(adj_matrix, unit_labels, top_n=15):
    """
    Identifies 'High Entropy' units - words that appear in many different contexts.
    These are often function words or highly polysemous words.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS 2: UNIT VERSATILITY (High Entropy)")
    print(f"Which units appear in the most diverse array of contexts?")
    print(f"{'='*60}")

    # Calculate entropy for each row (unit) distribution across clusters
    # Add small epsilon to avoid log(0)
    row_sums = np.sum(adj_matrix, axis=1, keepdims=True) + 1e-9
    probs = adj_matrix / row_sums
    unit_entropies = entropy(probs, axis=1)

    # Sort by entropy
    sorted_indices = np.argsort(unit_entropies)[::-1]
    
    print(f"{'Rank':<5} | {'Unit':<20} | {'Entropy':<8} | {'Freq':<8}")
    print("-" * 50)
    
    count = 0
    for idx in sorted_indices:
        if row_sums[idx] < 5: continue # Skip rare words to avoid noise
        
        unit_str = unit_labels[idx]
        ent = unit_entropies[idx]
        freq = row_sums[idx][0]
        
        print(f"{count+1:<5} | {unit_str:<20} | {ent:.4f}   | {int(freq):<8}")
        count += 1
        if count >= top_n: break

def automated_parse_test(parser, text_corpus, num_samples=5):
    """
    Splits the corpus into sentences and parses a random subset.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS 3: AUTOMATED PARSING SAMPLES")
    print(f"Parsing random sentences from the input...")
    print(f"{'='*60}")

    # Crude sentence splitter
    sentences = text_corpus.replace("?", ".").replace("!", ".").split(".")
    sentences = [s.strip() for s in sentences if len(s.split()) > 3] # Filter short stuff
    
    if not sentences:
        print("No valid sentences found to parse.")
        return

    # Sample random sentences
    samples = random.sample(sentences, min(num_samples, len(sentences)))
    
    for i, sent in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: '{sent}'")
        tokens = sent.split()
        
        try:
            result = parser.parse(tokens)
            parser.print_tree(result)
        except Exception as e:
            print(f"Parse Failed: {e}")

def main():
    input_file = "input.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # --- 1. CONFIGURATION ---
    # Tweak these for "Deep Analysis" mode
    data_loader.NUM_CLUSTERS = 256
    data_loader.MIN_FREQ = 3
    
    # --- 2. LOAD DATA ---
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text_corpus = f.read()
    
    # Truncate for speed if necessary, or process full
    # text_corpus = text_corpus[:100000] 

    print("Processing corpus (this may take a minute)...")
    processor = CorpusProcessor()
    processor.ingest_text(text_corpus)
    
    try:
        processor.build_jax_structures()
    except ValueError as e:
        print(e)
        return

    data_pack = processor.export()
    
    # Extract raw numpy arrays for analysis (no JAX needed for stat analysis)
    adj_matrix = np.array(data_pack['adj_matrix']) # (Units, Clusters)
    unit_labels = data_pack['unit_labels']         # ID -> Str

    # --- 3. RUN ANALYTICS ---
    analyze_clusters(adj_matrix, unit_labels)
    analyze_unit_versatility(adj_matrix, unit_labels)
    
    # --- 4. RUN PARSER CHECK ---
    print("\nInitializing Parser for sanity check...")
    parser = JAXParser(data_pack)
    automated_parse_test(parser, text_corpus)

if __name__ == "__main__":
    main()
