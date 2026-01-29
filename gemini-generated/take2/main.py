from catalog import AggregatedCatalog
from expansion import ExpansionEngine
from parser_v3 import TopDownParser

def main():
    # 1. Initialize
    catalog = AggregatedCatalog()
    
    # 2. Load Data (Reading your input.txt)
    print("Loading Corpus...")
    with open("input.txt", "r") as f:
        text = f.read()
    
    print("Building Catalog (Counting contexts)...")
    catalog.ingest(text)
    
    # 3. Initialize Engine
    engine = ExpansionEngine(catalog)
    parser = TopDownParser(engine)
    
    # 4. Interactive Loop
    print("\nSystem Ready. (Pure Python - No Embeddings)")
    print("Note: This relies entirely on exact word matches in contexts.")
    
    while True:
        sent = input("\n>> ")
        if sent == 'q': break
        
        tokens = catalog.tokenizer.tokenize(sent)
        tree = parser.parse(tokens)
        parser.print_tree(tree)

if __name__ == "__main__":
    main()
