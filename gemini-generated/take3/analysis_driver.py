import sys
import random
import math
from collections import Counter
from tokenizer import SimpleTokenizer
from catalog import AggregatedCatalog
from expansion import ExpansionEngine
from parser_v3 import TopDownParser

def analyze_context_slots(catalog, top_n=10):
    """
    Finds the most 'productive' contexts (where many different units fit).
    This reveals the grammatical 'slots' of the language.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS 1: PRODUCTIVE GRAMMATICAL SLOTS")
    print(f"Contexts that accept the highest variety of units")
    print(f"{'='*60}")
    
    # Count unique units per context
    context_productivity = {}
    for ctx, units in catalog.context_to_units.items():
        # productivity = number of unique units seen in this slot
        context_productivity[ctx] = len(units)
        
    # Sort by productivity
    sorted_ctx = sorted(context_productivity.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    for (left, right), count in sorted_ctx:
        # Show examples of what fits there
        examples = list(catalog.context_to_units[(left, right)])[:5]
        ex_str = ", ".join([" ".join(u) for u in examples])
        print(f"Slot: '{left} ___ {right}' | Unique Fillers: {count}")
        print(f"  Examples: {ex_str}, ...")
        print("-" * 40)

def analyze_unit_entropy(catalog, top_n=15):
    """
    Finds units that appear in the most diverse set of contexts.
    High entropy = Function words / Connectors.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS 2: UNIT VERSATILITY (High Entropy)")
    print(f"Units that function in many different environments")
    print(f"{'='*60}")
    
    unit_entropy = []
    
    for unit, contexts in catalog.unit_to_contexts.items():
        freq = catalog.unit_counts[unit]
        if freq < 10: continue # Skip rare stuff
        
        # Calculate entropy of context distribution
        ctx_counts = Counter(contexts)
        total = sum(ctx_counts.values())
        ent = 0
        for count in ctx_counts.values():
            p = count / total
            ent -= p * math.log(p)
            
        unit_entropy.append((unit, ent, freq))
        
    # Sort
    sorted_units = sorted(unit_entropy, key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"{'Rank':<5} | {'Unit':<20} | {'Entropy':<8} | {'Freq':<8}")
    print("-" * 50)
    for i, (unit, ent, freq) in enumerate(sorted_units):
        u_str = " ".join(unit)
        print(f"{i+1:<5} | {u_str:<20} | {ent:.4f}   | {freq:<8}")

def automated_parse_test(parser, catalog, num_samples=5):
    print(f"\n{'='*60}")
    print(f"ANALYSIS 3: PARSING SAMPLES")
    print(f"{'='*60}")
    
    # Try to parse actual lines from input.txt
    try:
        with open("input.txt", "r") as f:
            lines = [line.strip() for line in f if len(line.split()) > 4]
    except FileNotFoundError:
        print("input.txt not found.")
        return

    if not lines:
        print("No valid lines in input.txt")
        return

    samples = random.sample(lines, min(len(lines), num_samples))
    
    for i, sent in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: '{sent}'")
        tokens = catalog.tokenizer.tokenize(sent)
        
        try:
            tree = parser.parse(tokens)
            parser.print_tree(tree)
        except RecursionError:
            print("[Error: Sentence too long for recursion depth]")
        except Exception as e:
            print(f"[Error: {e}]")

def main():
    # 1. Setup
    catalog = AggregatedCatalog()
    
    # 2. Ingest
    try:
        with open("input.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: input.txt not found in directory.")
        return

    catalog.ingest(text)
    
    # 3. Analyze Patterns
    analyze_context_slots(catalog)
    analyze_unit_entropy(catalog)
    
    # 4. Test Parser
    engine = ExpansionEngine(catalog)
    parser = TopDownParser(engine)
    automated_parse_test(parser, catalog)

if __name__ == "__main__":
    main()
