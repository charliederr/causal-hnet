import os
import sys
import data_loader  # Import the module to adjust settings
from data_loader import CorpusProcessor
from parser_v2 import JAXParser

def main():
    input_file = "input.txt"
    
    # 1. Check for Input File
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found.")
        print("Please create this file and paste your text corpus into it.")
        return

    # 2. Configuration for Larger Data
    # For a large file, 64 clusters is too small. We bump it up here.
    # (We are modifying the global variable in the imported module)
    data_loader.NUM_CLUSTERS = 256  
    data_loader.MIN_FREQ = 3        # Ignore words that appear less than 3 times
    
    print(f"--- Configuration ---")
    print(f"Input: {input_file}")
    print(f"Clusters: {data_loader.NUM_CLUSTERS}")
    print(f"Min Freq: {data_loader.MIN_FREQ}")

    # 3. Load and Process Data
    print(f"\nReading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        text_corpus = f.read()

    if len(text_corpus.strip()) == 0:
        print("Error: Input file is empty.")
        return

    print(f"Ingesting corpus ({len(text_corpus)} characters)...")
    processor = CorpusProcessor()
    processor.ingest_text(text_corpus)
    
    print("Building JAX structures (Clustering)...")
    try:
        processor.build_jax_structures()
    except ValueError as e:
        print(f"\nError during clustering: {e}")
        print("Tip: Your input text might be too small for the requested number of clusters.")
        return

    # Export the "Catalog" (Centroids + Adjacency Matrix)
    data_pack = processor.export()

    # 4. Initialize Parser with REAL Data
    print("\nInitializing Parser with corpus data...")
    parser = JAXParser(data_pack)

    # 5. Interactive Loop
    print("\n" + "="*40)
    print("      SYSTEM READY      ")
    print("Type a sentence to parse (or 'q' to quit)")
    print("="*40)
    
    while True:
        try:
            user_input = input("\n>> ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            tokens = user_input.split()
            if not tokens: continue
            
            # Run the parser
            result = parser.parse(tokens)
            
            # Print the tree
            print("\n[Parse Tree]")
            parser.print_tree(result)
            
        except KeyboardInterrupt:
            print("\nQuitting...")
            break
        except Exception as e:
            print(f"Error parsing sentence: {e}")

if __name__ == "__main__":
    main()
