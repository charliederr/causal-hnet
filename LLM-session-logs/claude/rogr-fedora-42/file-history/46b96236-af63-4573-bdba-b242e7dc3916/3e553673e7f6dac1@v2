# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository explores causal inference and attention mechanisms in neural networks, building on research papers available at https://nonlanguage.dev/causal-hnet/papers/. The project implements experimental NLP pipelines combining:

- Soft clustering via mean-field inference with simulated annealing
- Freeman-style attention templates with directional co-occurrence coupling
- Event-local template assignment with sparse top-m assignment
- Character n-gram hashed word embeddings
- Hierarchical parsing via attention-based constituency induction

## Running the Experiments

The Python scripts require JAX and NumPy. All scripts expect an `input.txt` file in the working directory (a text corpus for processing).

```bash
# Run the main soft-constraint attention template experiment
cd causal-hnet/chatgpt-generated
python soft_constraints_hnet_v7.py

# Run the event-local template variant (supports CLI args)
python soft_constraints_attention_hierarchy_eventlocal_v4.py --input input.txt --clusters 16 --n_segs 6
```

Key CLI arguments for the event-local scripts:
- `--input`: path to input text file
- `--clusters`: number of template clusters (default 16)
- `--seg_len`: attention segment length (default 256)
- `--n_segs`: number of segments to sample
- `--topm_templates`: sparse template assignment (default 2)
- `--topk_attn`: sparse attention per row

## Directory Structure

- `causal-hnet/chatgpt-generated/`: Main implementation scripts (JAX-based)
- `causal-hnet/claude-generated/`: Alternative implementations
- `docs/`: Reference PDFs (Causal-HNet paper, Freeman-Transformers papers)
- `causal-hnet/LLM-session-logs/`: Session logs from development

## Architecture Notes

The core pipeline in the v7/v4 scripts:
1. **Vocabulary construction**: Frequency-filtered tokens with IDF weighting
2. **Word embeddings**: Character n-gram hashing into fixed buckets, averaged per word
3. **Template clustering**: Mean-field soft assignment with annealing (τ schedule), directional co-occurrence edges, and size penalties
4. **Attention computation**: QK dot-product attention biased by template compatibility (W_psi), directional J_pmi coupling, and IDF-based target weighting
5. **Hierarchy extraction**: Binary bracket parsing via cross-attention mass minimization

The J_pmi matrix captures directional cluster-to-cluster transition preferences learned from co-occurrence statistics. Template biases modulate attention to prefer semantically compatible token pairs.
