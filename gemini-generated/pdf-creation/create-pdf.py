import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

def create_plan_pdf():
    filename = "jax_parsing_plan.pdf"
    
    # Configuration for text layout
    left_margin = 0.1
    top_margin = 0.9
    line_height = 0.035  # Slightly tighter to fit more text
    
    # Create the PDF object
    pdf = PdfPages(filename)
    
    # --- Helper Function to Add a Page ---
    def add_page(title, lines):
        fig = plt.figure(figsize=(8.5, 11)) # Standard Letter size
        
        # Add Title
        plt.text(left_margin, 0.95, title, fontsize=16, weight='bold', color='#2c3e50')
        plt.axhline(y=0.93, xmin=0.05, xmax=0.95, color='black', linewidth=1)
        
        # Add Text Content
        y_pos = top_margin
        for line in lines:
            # Check if line is a section header (starts with ##)
            if line.startswith("##"):
                y_pos -= line_height * 1.5
                plt.text(left_margin, y_pos, line.replace("##", "").strip(), 
                         fontsize=12, weight='bold', color='#2c3e50')
                y_pos -= line_height
            # Check if line is a bullet point
            elif line.strip().startswith("*"):
                # Wrap text slightly narrower for bullets
                wrapped = textwrap.wrap(line.strip(), width=85)
                for w in wrapped:
                    plt.text(left_margin + 0.02, y_pos, w, fontsize=10, fontname='DejaVu Sans')
                    y_pos -= line_height
            # Check if line is math (starts with $)
            elif line.strip().startswith("$"):
                plt.text(left_margin + 0.05, y_pos, line.strip(), fontsize=12, style='italic', color='#8e44ad')
                y_pos -= line_height * 1.5
            # Standard text
            else:
                wrapped = textwrap.wrap(line.strip(), width=90)
                for w in wrapped:
                    plt.text(left_margin, y_pos, w, fontsize=10, fontname='DejaVu Sans')
                    y_pos -= line_height
            
            # Simple page break check
            if y_pos < 0.1:
                plt.text(0.5, 0.05, "(Cont...)", ha='center', fontsize=8)
                break
                
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()

    # --- CONTENT DEFINITION ---
    # Note: These strings summarize the plan based on the parsing_approach_summary.pdf
    
    page_1_content = [
        "This document outlines the plan for implementing a JAX-accelerated parser based on",
        "bidirectional context expansion.",
        "",
        "## Phase 1: Data Representation & 'The Catalog'",
        "To make the 'Catalog' (currently 32k n-grams) GPU-compatible, we must vectorize it.",
        "",
        "## 1.1 Context Vectorization",
        "* Instead of storing tuple strings (left, right), we encode them into vectors.",
        "* Approach: Use a lightweight embedding (e.g., small BERT/GloVe) for context windows.",
        "* Representation: A context c becomes a vector v_ctx.",
        "* Benefit: Solves the representation challenge regarding pattern generalization.",
        "",
        "## 1.2 The Interaction Matrix (Sparse to Dense)",
        "We construct a global map of Units fitting into Contexts.",
        "* Construct a frequency-weighted Adjacency Matrix A.",
        "* Rows = Unique Units (e.g., 'go out', 'my money').",
        "* Cols = Context Cluster Centroids (e.g., k=4096).",
        "* A_ij = 1 (or freq) if Unit i appears in Context Cluster j.",
        "* JAX Optimization: Store A as a sparse matrix (BCOO) on GPU.",
        "",
        "## Phase 2: The Expansion Kernel (JAX Core)",
        "We replace iterative loops with vectorized similarity propagation.",
        "",
        "## 2.1 Context Expansion (The 'Where else?')",
        "* Input: Vector of current context v_ctx.",
        "* Operation: Cosine Similarity between v_ctx and Context Centroids.",
        "* Output: A 'soft mask' vector m_ctx where values are similarity scores.",
    ]

    page_2_content = [
        "## 2.2 Unit Expansion (The 'Who else?')",
        "* Operation: Multiply the context mask m_ctx by the Adjacency Matrix A.",
        "* Math:",
        "$v_{units} = A \\cdot m_{ctx}$",
        "* Result: A vector where high values indicate units fitting the expanded contexts.",
        "",
        "## 2.3 Energy Calculation",
        "* We implement the energy score using differentiable ops:",
        "$E = -\\log(|units| \\times \\log(|contexts| + 1))$",
        "* Note: We use Softmax/LogSumExp to approximate counts for differentiability.",
        "",
        "## Phase 3: The Parsing Shell (Hybrid)",
        "## 3.1 Top-Down Controller (Python)",
        "* Manages recursive splitting of the sentence.",
        "* Identifies candidate spans (e.g., 'my money').",
        "",
        "## 3.2 Batch Scoring (JAX)",
        "* Collect all candidate spans for a sentence.",
        "* Send them to GPU in one batch to compute Expansion Energy simultaneously.",
        "",
        "## 3.3 Ambiguity Resolution",
        "* Compare unit_energy vs best_split_energy to decide between unit or split.",
        "",
        "## Phase 4: Specific Challenges",
        "* Polysemy (e.g., 'go out'): Preserved by the input context vector v_ctx generating",
        "  distinct masks m_ctx for social vs. motion contexts.",
        "* Scalability: Solved by JAX matrix ops, avoiding O(n^2) comparisons.",
    ]
    
    page_3_content = [
        "## Suggested Module Structure",
        "",
        "1. data_loader.py",
        "* Ingests text and builds Unit-Context counts.",
        "* Runs K-Means to compress raw contexts into centroids.",
        "",
        "2. expansion_engine.py (JAX)",
        "* Contains @jax.jit functions for similarity.",
        "* Holds static matrices (Adjacency A, Centroids C) in GPU memory.",
        "",
        "3. parser.py",
        "* Implements recursive logic and batches spans for the engine.",
        "",
        "## Next Steps",
        "We will begin by mocking up the Expansion Engine to demonstrate the vector math",
        "before building the data loader."
    ]

    # Add the pages
    add_page("Project Plan: JAX-Accelerated Bidirectional Parsing", page_1_content)
    add_page("JAX Expansion & Parsing Logic", page_2_content)
    add_page("Architecture & Modules", page_3_content)

    pdf.close()
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    create_plan_pdf()
