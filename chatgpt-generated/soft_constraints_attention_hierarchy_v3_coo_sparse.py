import argparse
import math
import random
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================
# Utilities
# ============================================================

_punct_re = re.compile(r"^[\W_]+$", re.UNICODE)

def is_punct_heavy(token: str) -> bool:
    if _punct_re.match(token):
        return True
    alnum = sum(ch.isalnum() for ch in token)
    return alnum <= 1 and len(token) <= 4

def normalize_for_ngrams(token: str) -> str:
    t = token.lower()
    t = re.sub(r"^[^a-z0-9]+", "", t)
    t = re.sub(r"[^a-z0-9]+$", "", t)
    return t

def char_ngrams(s: str, nmin: int, nmax: int):
    if not s:
        return []
    s2 = f"<{s}>"
    out = []
    L = len(s2)
    for n in range(nmin, nmax + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            out.append(s2[i:i+n])
    return out

def stable_hash32(x: str) -> int:
    h = 2166136261
    for ch in x:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def fmt_templates(p: np.ndarray, k: int = 3) -> str:
    idx = np.argsort(-p)[:k]
    return " ".join([f"t{int(i):02d}:{p[int(i)]:.2f}" for i in idx])

# ============================================================
# Hierarchy helpers (cross-mass from sparse edges)
# ============================================================

def build_dense_from_edges(S: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    A = np.zeros((S, S), dtype=np.float32)
    A[src, dst] = w
    return A

def best_split_by_cross_mass_dense(A: np.ndarray, i: int, j: int) -> Tuple[int, float]:
    best_k = i
    best_score = float("inf")
    for k in range(i, j):
        if k + 1 > j:
            continue
        left = slice(i, k + 1)
        right = slice(k + 1, j + 1)
        cross = A[left, right].sum() + A[right, left].sum()
        if cross < best_score:
            best_score = float(cross)
            best_k = k
    return best_k, best_score

def build_parse_dense(A: np.ndarray, tokens: List[str], i: int, j: int, depth: int = 0, max_depth: int = 6) -> str:
    if i == j or depth >= max_depth:
        return tokens[i]
    k, _ = best_split_by_cross_mass_dense(A, i, j)
    if k >= j:
        return tokens[i]
    left = build_parse_dense(A, tokens, i, k, depth + 1, max_depth)
    right = build_parse_dense(A, tokens, k + 1, j, depth + 1, max_depth)
    return f"({left} {right})"


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--max_vocab", type=int, default=20000)
    ap.add_argument("--cooc_window", type=int, default=6)

    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--tau_start", type=float, default=1.25)
    ap.add_argument("--tau_end", type=float, default=0.12)
    ap.add_argument("--damping", type=float, default=0.35)
    ap.add_argument("--gamma_size", type=float, default=1.25)
    ap.add_argument("--eta_unary", type=float, default=0.75)

    ap.add_argument("--emb_dim", type=int, default=96)
    ap.add_argument("--head_dim", type=int, default=64)

    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--hash_buckets", type=int, default=2**18)
    ap.add_argument("--ngram_scale", type=float, default=0.2)

    ap.add_argument("--seg_len", type=int, default=256)
    ap.add_argument("--n_segs", type=int, default=6)
    ap.add_argument("--attn_local_window", type=int, default=64)
    ap.add_argument("--token_noise", type=float, default=0.01)

    ap.add_argument("--alpha_template", type=float, default=1.0)
    ap.add_argument("--beta_directional", type=float, default=0.75)
    ap.add_argument("--lambda_idf_target", type=float, default=0.15)
    ap.add_argument("--lambda_antihub", type=float, default=0.35)

    ap.add_argument("--topk_attn", type=int, default=32)

    ap.add_argument("--top_edges", type=int, default=1000)
    ap.add_argument("--top_edges_with_clusters", type=int, default=100)

    ap.add_argument("--min_idf_print", type=float, default=1.8)
    ap.add_argument("--skip_self_links", action="store_true", default=True)
    ap.add_argument("--skip_punct_heavy", action="store_true", default=True)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().lower()
    raw_tokens = text.split()
    T = len(raw_tokens)

    print("=" * 80)
    print("HNET — COO GRAPH k-SPARSE ATTENTION + ANTI-HUB + HIERARCHY (SOFT TEMPLATES) v3")
    print("=" * 80)
    print(f"Total tokens: {T}")

    # vocab
    freq = Counter(raw_tokens)
    vocab = [w for w, c in freq.items() if c >= args.min_freq]
    vocab = vocab[:args.max_vocab]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"Vocabulary size: {V}")

    tok_ids = np.full((T,), -1, dtype=np.int32)
    for i, w in enumerate(raw_tokens):
        tok_ids[i] = word_to_id.get(w, -1)

    # IDF
    counts_vocab = np.zeros(V, dtype=np.int64)
    for tid in tok_ids:
        if tid >= 0:
            counts_vocab[tid] += 1
    N = counts_vocab.sum()
    idf = np.log((N + 1.0) / (counts_vocab + 1.0)) + 1.0
    idf = idf.astype(np.float32)
    idf_j = jnp.array(idf)

    # cooc edges
    cooc = defaultdict(float)
    for i in range(T):
        wi = tok_ids[i]
        if wi < 0:
            continue
        end = min(i + args.cooc_window, T)
        for j in range(i + 1, end):
            wj = tok_ids[j]
            if wj < 0:
                continue
            cooc[(wi, wj)] += 1.0

    norm_out = np.zeros(V, dtype=np.float64)
    norm_in = np.zeros(V, dtype=np.float64)
    for (i, j), v in cooc.items():
        norm_out[i] += v
        norm_in[j] += v

    edges_i, edges_j, edges_w = [], [], []
    for (i, j), v in cooc.items():
        w = v / math.sqrt(norm_out[i] * norm_in[j] + 1e-8)
        edges_i.append(i); edges_j.append(j); edges_w.append(w)

    edges_i = jnp.array(edges_i, dtype=jnp.int32)
    edges_j = jnp.array(edges_j, dtype=jnp.int32)
    edges_w = jnp.array(edges_w, dtype=jnp.float32)

    print(f"Directional edges (word->word): {int(edges_i.shape[0])}")

    # ngram embeddings
    rng = np.random.RandomState(args.seed)
    bucket_vecs = rng.normal(size=(args.hash_buckets, args.emb_dim)).astype(np.float32)
    bucket_vecs *= (args.ngram_scale / math.sqrt(args.emb_dim))

    Xw_np = np.zeros((V, args.emb_dim), dtype=np.float32)
    for wid, token in enumerate(vocab):
        normed = normalize_for_ngrams(token)
        ngs = char_ngrams(normed, args.ngram_min, args.ngram_max)
        if not ngs:
            Xw_np[wid] = rng.normal(size=(args.emb_dim,)).astype(np.float32) * 0.01
            continue
        idxs = [stable_hash32(g) % args.hash_buckets for g in ngs]
        Xw_np[wid] = bucket_vecs[idxs].mean(axis=0)

    Xw = jnp.array(Xw_np)

    # template prototypes
    key = jax.random.PRNGKey(args.seed)
    key, subz = jax.random.split(key)
    Zc = jax.random.normal(subz, (args.clusters, args.emb_dim)) * 0.5
    UNARY = args.eta_unary * (Xw @ Zc.T)

    Zc_norm = Zc / (jnp.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
    W_psi = Zc_norm @ Zc_norm.T

    # soft clustering
    def tau_at(sweep: int) -> float:
        t = sweep / max(1, (args.sweeps - 1))
        return args.tau_start * ((args.tau_end / args.tau_start) ** t)

    @jax.jit
    def mean_entropy(P):
        return -jnp.mean(jnp.sum(P * jnp.log(P + 1e-12), axis=1))

    @jax.jit
    def soft_sweep(P, tau):
        contrib = edges_w[:, None] * P[edges_j]
        field = jnp.zeros((V, args.clusters), dtype=P.dtype).at[edges_i].add(contrib)

        counts = jnp.sum(P, axis=0) + 1.0
        penalty = args.gamma_size * jnp.log(counts)[None, :]

        score = field + UNARY - penalty
        P_new = jax.nn.softmax(score / tau, axis=1)

        P_mixed = (1.0 - args.damping) * P + args.damping * P_new
        P_mixed = P_mixed / (jnp.sum(P_mixed, axis=1, keepdims=True) + 1e-12)
        return P_mixed

    P = jnp.ones((V, args.clusters), dtype=jnp.float32) / float(args.clusters)
    print("Running SOFT (mean-field + anneal) directional sweeps...")
    for s in range(args.sweeps):
        tau = tau_at(s)
        P = soft_sweep(P, tau)
        if (s + 1) % 5 == 0 or s == 0:
            counts_soft = np.array(jnp.sum(P, axis=0))
            ent = float(mean_entropy(P))
            hard = np.array(jnp.argmax(P, axis=1))
            counts_hard = np.bincount(hard, minlength=args.clusters)
            print(
                f"  sweep {s+1}/{args.sweeps}  tau={tau:.3f}  mean_entropy={ent:.3f}  "
                f"soft_counts(min/med/max)=({counts_soft.min():.1f}/{np.median(counts_soft):.1f}/{counts_soft.max():.1f})  "
                f"argmax_nonempty={int((counts_hard>0).sum())}"
            )

    P_np = np.array(P)

    # J_pmi
    Pi = P[edges_i]
    Pj = P[edges_j]
    Pi_w = Pi * edges_w[:, None]

    J_mass = (Pi_w.T @ Pj)
    J_row = J_mass / (jnp.sum(J_mass, axis=1, keepdims=True) + 1e-12)

    pi = jnp.sum(P, axis=0) / float(V)
    base = pi[:, None] * pi[None, :]
    J_pmi = jnp.log((J_row + 1e-12) / (base + 1e-12))
    J_pmi = J_pmi - jnp.mean(J_pmi)
    J_pmi = J_pmi / (jnp.std(J_pmi) + 1e-6)
    J_pmi_np = np.array(J_pmi)

    print("\n" + "-" * 80)
    print("TOP CLUSTER→CLUSTER DIRECTIONAL COUPLINGS (J_pmi)")
    print("-" * 80)
    pairs = []
    for a in range(args.clusters):
        for b in range(args.clusters):
            pairs.append((J_pmi_np[a, b], a, b))
    pairs.sort(reverse=True)
    for val, a, b in pairs[:20]:
        print(f"c{a:02d} → c{b:02d}   J_pmi={val:.3f}")

    # projections
    key, subq, subk = jax.random.split(key, 3)
    Wq = jax.random.normal(subq, (args.emb_dim, args.head_dim)) / math.sqrt(args.emb_dim)
    Wk = jax.random.normal(subk, (args.emb_dim, args.head_dim)) / math.sqrt(args.emb_dim)

    Kk = min(int(args.topk_attn), int(args.seg_len) - 1)
    NEG_LARGE = -1e9

    @jax.jit
    def segment_attention_edges(word_ids_seg, noise_key):
        """
        Return COO edge list for k-sparse attention:
          src: (S*Kk,)
          dst: (S*Kk,)
          w:   (S*Kk,)
        Also return dense bias matrices for diagnostics:
          base_logits, tpl_bias, dir_bias: (S,S)
        And pair_mass: (C,C).
        """
        Ew = Xw[word_ids_seg]
        psi = P[word_ids_seg]

        eps = args.token_noise * jax.random.normal(noise_key, Ew.shape)
        Eseg = Ew + (psi @ Zc) + eps

        Q = Eseg @ Wq
        K = Eseg @ Wk
        base_logits = (Q @ K.T) / jnp.sqrt(args.head_dim)

        S = word_ids_seg.shape[0]
        idx = jnp.arange(S)
        dist = jnp.abs(idx[:, None] - idx[None, :])
        local = dist <= args.attn_local_window

        tpl_bias = (psi @ W_psi) @ psi.T
        dir_bias = (psi @ J_pmi) @ psi.T

        idf_tgt = idf_j[word_ids_seg][None, :]
        hub_penalty = -args.lambda_antihub * (1.0 / (idf_tgt + 1e-6))

        logits = (
            base_logits
            + args.alpha_template * tpl_bias
            + args.beta_directional * dir_bias
            + args.lambda_idf_target * idf_tgt
            + hub_penalty
        )

        logits = jnp.where(local, logits, NEG_LARGE)
        logits = logits - 1e9 * jnp.eye(S, dtype=logits.dtype)

        # topk per row
        _, topk_idx = jax.lax.top_k(logits, Kk)  # (S,Kk)

        # softmax over those Kk entries per row, return weights
        def row_probs(i):
            idxs = topk_idx[i]
            sel = logits[i, idxs]
            sel = sel - jnp.max(sel)
            ex = jnp.exp(sel)
            probs = ex / (jnp.sum(ex) + 1e-12)
            return probs

        probs = jax.vmap(row_probs)(jnp.arange(S))  # (S,Kk)

        # COO edge list
        src = jnp.repeat(jnp.arange(S), Kk)               # (S*Kk,)
        dst = topk_idx.reshape(-1)                        # (S*Kk,)
        w = probs.reshape(-1)                             # (S*Kk,)

        pair_mass = psi.T @ (jnp.zeros((S, S), dtype=logits.dtype).at[src, dst].set(w)) @ psi
        return src, dst, w, base_logits, tpl_bias, dir_bias, pair_mass

    # compile
    starts = [random.randint(0, T - args.seg_len - 1) for _ in range(args.n_segs)]
    print("\nCompiling COO sparse attention (one-time JAX cost)...")
    dummy = tok_ids[starts[0]:starts[0] + args.seg_len].copy()
    dummy[dummy < 0] = 0
    _ = segment_attention_edges(jnp.array(dummy, dtype=jnp.int32), key)[0].block_until_ready()
    print("Compilation finished.\n")

    # template peers
    template_peers: Dict[int, List[int]] = {}
    for c in range(args.clusters):
        idx = np.argsort(-P_np[:, c])[:50]
        template_peers[c] = idx.tolist()

    def peers_for_template(tidx: int) -> str:
        ids = template_peers[int(tidx)][:10]
        return ", ".join([vocab[i] for i in ids])

    # collect edges across segments
    all_edges = []
    template_flow = np.zeros((args.clusters, args.clusters), dtype=np.float64)

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start:start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]
        seg_j = jnp.array(seg, dtype=jnp.int32)

        key, nk = jax.random.split(key)
        src, dst, w, baseL, tplB, dirB, pair_mass = segment_attention_edges(seg_j, nk)

        src_np = np.array(src)
        dst_np = np.array(dst)
        w_np = np.array(w)
        base_np = np.array(baseL)
        tpl_np = np.array(tplB)
        dir_np = np.array(dirB)

        template_flow += np.array(pair_mass)

        # record edges
        for s_idx, d_idx, ww in zip(src_np, dst_np, w_np):
            if s_idx == d_idx:
                continue
            wi = int(seg[s_idx])
            wj = int(seg[d_idx])

            if args.skip_self_links and wi == wj:
                continue
            if idf[wi] < args.min_idf_print or idf[wj] < args.min_idf_print:
                continue
            if args.skip_punct_heavy and (is_punct_heavy(seg_words[s_idx]) or is_punct_heavy(seg_words[d_idx])):
                continue

            all_edges.append({
                "seg": seg_i,
                "pos_i": start + int(s_idx),
                "pos_j": start + int(d_idx),
                "w_i": seg_words[int(s_idx)],
                "w_j": seg_words[int(d_idx)],
                "attn": float(ww),
                "score_idf": float(ww * idf[wi] * idf[wj]),
                "base": float(base_np[int(s_idx), int(d_idx)]),
                "tpl": float(tpl_np[int(s_idx), int(d_idx)]),
                "dir": float(dir_np[int(s_idx), int(d_idx)]),
                "idf_i": float(idf[wi]),
                "idf_j": float(idf[wj]),
                "dist": abs(int(s_idx) - int(d_idx)),
                "wid_i": wi,
                "wid_j": wj,
                "psi_i": P_np[wi].copy(),
                "psi_j": P_np[wj].copy(),
            })

    template_flow /= max(1, args.n_segs)

    # template flow reporting
    print("-" * 80)
    print("TEMPLATE→TEMPLATE ATTENTION FLOWS (aggregated over sampled segments)")
    print("-" * 80)
    pairs = []
    for a in range(args.clusters):
        for b in range(args.clusters):
            if a == b:
                continue
            pairs.append((template_flow[a, b], a, b))
    pairs.sort(reverse=True)
    for val, a, b in pairs[:16]:
        print(f"Template {a:02d} → Template {b:02d}   flow={val:.4f}   J_pmi={J_pmi_np[a,b]:+.3f}   Wpsi={float(W_psi[a,b]):.4f}")

    # top edges
    all_edges.sort(key=lambda r: r["score_idf"], reverse=True)
    topN = min(args.top_edges, len(all_edges))
    topM = min(args.top_edges_with_clusters, len(all_edges))

    print("\n" + "-" * 80)
    print(f"TOP {topN} TOKEN→TOKEN LINKS (IDF-weighted; COO k-sparse; anti-hub; filtered)")
    print("-" * 80)
    for r in all_edges[:topN]:
        near = "LOCAL" if r["dist"] <= args.attn_local_window else "LONG"
        print(
            f"[seg {r['seg']}] {near:>4}  "
            f"{r['w_i']:<14} → {r['w_j']:<14}  "
            f"score={r['score_idf']:.4f} (attn={r['attn']:.4f})  "
            f"base={r['base']:+.3f} tpl={args.alpha_template*r['tpl']:+.3f} dir={args.beta_directional*r['dir']:+.3f}  "
            f"idf=({r['idf_i']:.2f},{r['idf_j']:.2f}) dist={r['dist']}"
        )

    # cluster context topM
    print("\n" + "-" * 80)
    print(f"CLUSTER CONTEXT FOR TOP {topM} LINKS (template weights + peers)")
    print("-" * 80)

    for rank, r in enumerate(all_edges[:topM], 1):
        psi_i = r["psi_i"]
        psi_j = r["psi_j"]
        top_i = np.argsort(-psi_i)[:3]
        top_j = np.argsort(-psi_j)[:3]

        print(f"\n#{rank:03d} [seg {r['seg']}] {r['w_i']} → {r['w_j']}  score={r['score_idf']:.4f} attn={r['attn']:.4f} dist={r['dist']}")
        print(f"  src templates: {fmt_templates(psi_i, 3)}")
        for t in top_i[:2]:
            print(f"    src peers for t{int(t):02d}: {peers_for_template(int(t))}")

        print(f"  tgt templates: {fmt_templates(psi_j, 3)}")
        for t in top_j[:2]:
            print(f"    tgt peers for t{int(t):02d}: {peers_for_template(int(t))}")

    # hierarchy parses
    print("\n" + "-" * 80)
    print("HIERARCHY (binary bracket parses) for sampled segments")
    print("-" * 80)

    # For parsing we build a dense A from COO edges (S=256, cheap) per segment.
    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start:start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]
        seg_j = jnp.array(seg, dtype=jnp.int32)

        key, nk = jax.random.split(key)
        src, dst, w, *_ = segment_attention_edges(seg_j, nk)

        src_np = np.array(src)
        dst_np = np.array(dst)
        w_np = np.array(w)

        A = build_dense_from_edges(args.seg_len, src_np, dst_np, w_np)

        L = min(64, args.seg_len)
        parse = build_parse_dense(A[:L, :L], seg_words[:L], 0, L - 1, max_depth=6)
        print(f"\n[seg {seg_i}] tokens {start}..{start+L-1}")
        print(parse)

    print("\nDone.")


if __name__ == "__main__":
    main()
