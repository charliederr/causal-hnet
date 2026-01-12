#!/usr/bin/env python3
"""
Preprocessing script to generate input files for hdp1.py (Step 4B parser).

Generates:
1. hnet_tokens_vocab.npz - tokenized corpus + vocabulary
2. hnet_dir_edges.npz - directional word->word co-occurrence edges
3. hnet_q0.npz - optional global role priors

Usage:
    python preprocess_corpus.py --input input.txt --K 16
"""

import argparse
import numpy as np
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="input.txt",
                    help="Input text file")
    ap.add_argument("--min_freq", type=int, default=5,
                    help="Minimum word frequency")
    ap.add_argument("--max_vocab", type=int, default=20000,
                    help="Maximum vocabulary size")
    ap.add_argument("--cooc_window", type=int, default=6,
                    help="Co-occurrence window size (directional)")
    ap.add_argument("--K", type=int, default=16,
                    help="Number of role templates (for q0 initialization)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed")
    args = ap.parse_args()

    print("=" * 80)
    print("PREPROCESSING CORPUS FOR HAMILTONIAN DYNAMIC PARSER")
    print("=" * 80)

    # Read and tokenize
    print(f"\nReading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().lower()

    tokens = text.split()
    T = len(tokens)
    print(f"Total tokens: {T:,}")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    freq = Counter(tokens)
    vocab_list = [w for w, c in freq.most_common() if c >= args.min_freq]
    vocab_list = vocab_list[:args.max_vocab]
    word_to_id = {w: i for i, w in enumerate(vocab_list)}
    V = len(vocab_list)
    print(f"Vocabulary size: {V:,} (min_freq={args.min_freq})")

    # Convert to token IDs
    print("Converting tokens to IDs...")
    tok_ids = []
    oov_count = 0
    for w in tokens:
        if w in word_to_id:
            tok_ids.append(word_to_id[w])
        else:
            oov_count += 1

    tok_ids = np.array(tok_ids, dtype=np.int32)
    print(f"Token IDs: {len(tok_ids):,} (OOV dropped: {oov_count:,})")

    # Build directional co-occurrence edges
    print(f"\nBuilding directional edges (window={args.cooc_window})...")
    cooc = defaultdict(float)

    for i in range(len(tok_ids)):
        wi = tok_ids[i]
        end = min(i + args.cooc_window + 1, len(tok_ids))
        for j in range(i + 1, end):
            wj = tok_ids[j]
            cooc[(wi, wj)] += 1.0

    # Convert to COO format
    src_list = []
    dst_list = []
    w_list = []

    for (i, j), count in cooc.items():
        src_list.append(i)
        dst_list.append(j)
        w_list.append(count)

    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    w = np.array(w_list, dtype=np.float32)

    print(f"Directional edges: {len(src):,}")

    # Compute PMI-like weights
    print("Computing PMI weights...")
    word_counts = np.bincount(tok_ids, minlength=V)
    total_pairs = w.sum()

    pmi = np.zeros_like(w)
    for idx in range(len(src)):
        i, j = src[idx], dst[idx]
        p_ij = w[idx] / total_pairs
        p_i = word_counts[i] / len(tok_ids)
        p_j = word_counts[j] / len(tok_ids)
        pmi[idx] = np.log(p_ij / (p_i * p_j + 1e-12) + 1e-12)

    # Normalize weights: positive PMI
    pmi = np.maximum(pmi, 0.0)
    w_pmi = pmi.astype(np.float32)

    # Initialize q0 (uniform with small noise)
    print(f"\nInitializing q0 role priors (K={args.K})...")
    rng = np.random.default_rng(args.seed)
    q0 = np.ones((V, args.K), dtype=np.float32) / float(args.K)
    q0 += 1e-3 * rng.standard_normal((V, args.K)).astype(np.float32)
    q0 = np.clip(q0, 1e-6, None)
    q0 /= q0.sum(axis=-1, keepdims=True)

    # Save files
    print("\nSaving output files...")

    # 1. Tokens + vocab
    np.savez_compressed(
        "hnet_tokens_vocab.npz",
        tokens=tok_ids,
        vocab=np.array(vocab_list, dtype=object)
    )
    print("  ✓ hnet_tokens_vocab.npz")

    # 2. Directional edges
    np.savez_compressed(
        "hnet_dir_edges.npz",
        src=src,
        dst=dst,
        w=w_pmi,
        w_raw=w
    )
    print("  ✓ hnet_dir_edges.npz")

    # 3. Role priors
    np.savez_compressed(
        "hnet_q0.npz",
        q0=q0
    )
    print("  ✓ hnet_q0.npz")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("\nYou can now run the parser:")
    print(f"  python hdp1.py --K {args.K} --q0_path hnet_q0.npz")
    print()


if __name__ == "__main__":
    main()
