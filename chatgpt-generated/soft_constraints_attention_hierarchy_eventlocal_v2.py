import argparse
import math
import random
import re
from collections import Counter
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
# Hierarchy parse (dense on first 64 tokens)
# ============================================================

def best_split_by_cross_mass(A: np.ndarray, i: int, j: int) -> Tuple[int, float]:
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

def build_parse(A: np.ndarray, tokens: List[str], i: int, j: int, depth: int = 0, max_depth: int = 6) -> str:
    if i == j or depth >= max_depth:
        return tokens[i]
    k, _ = best_split_by_cross_mass(A, i, j)
    if k >= j:
        return tokens[i]
    left = build_parse(A, tokens, i, k, depth + 1, max_depth)
    right = build_parse(A, tokens, k + 1, j, depth + 1, max_depth)
    return f"({left} {right})"


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", type=str, default="input.txt")
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--max_vocab", type=int, default=20000)

    # event-local cooc window only used for anti-hub proxy (optional); main dir now comes from attention edges
    ap.add_argument("--cooc_window", type=int, default=6)

    # templates
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--emb_dim", type=int, default=96)

    # event-local psi
    ap.add_argument("--tau_psi", type=float, default=0.35)
    ap.add_argument("--gamma_ctx", type=float, default=0.6)     # context strength
    ap.add_argument("--eta_psidot", type=float, default=0.7)    # NEW: adds psi·psi similarity into tpl_bias

    # embeddings (ngram hashed)
    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--hash_buckets", type=int, default=2**18)
    ap.add_argument("--ngram_scale", type=float, default=0.2)

    # attention
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--seg_len", type=int, default=256)
    ap.add_argument("--n_segs", type=int, default=6)
    ap.add_argument("--attn_local_window", type=int, default=64)
    ap.add_argument("--token_noise", type=float, default=0.01)

    # bias weights
    ap.add_argument("--alpha_template", type=float, default=1.0)
    ap.add_argument("--beta_directional", type=float, default=0.75)
    ap.add_argument("--lambda_idf_target", type=float, default=0.15)
    ap.add_argument("--lambda_antihub", type=float, default=0.35)

    # sparse attention
    ap.add_argument("--topk_attn", type=int, default=32)

    # reporting
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

    # ------------------------------------------------------------
    # Load text
    # ------------------------------------------------------------
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read().lower()

    raw_tokens = text.split()
    T = len(raw_tokens)

    print("=" * 80)
    print("HNET — EVENT-LOCAL TEMPLATES v2 (DIR from inferred ATTENTION) + COO k-SPARSE + HIERARCHY")
    print("=" * 80)
    print(f"Total tokens: {T}")

    # ------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------
    freq = Counter(raw_tokens)
    vocab = [w for w, c in freq.items() if c >= args.min_freq]
    vocab = vocab[:args.max_vocab]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"Vocabulary size: {V}")

    tok_ids = np.full((T,), -1, dtype=np.int32)
    for i, w in enumerate(raw_tokens):
        tok_ids[i] = word_to_id.get(w, -1)

    # ------------------------------------------------------------
    # IDF over token stream
    # ------------------------------------------------------------
    counts_vocab = np.zeros(V, dtype=np.int64)
    for tid in tok_ids:
        if tid >= 0:
            counts_vocab[tid] += 1
    N = counts_vocab.sum()
    idf = np.log((N + 1.0) / (counts_vocab + 1.0)) + 1.0
    idf = idf.astype(np.float32)
    idf_j = jnp.array(idf)

    # ------------------------------------------------------------
    # Build n-gram hashed word embeddings (CPU -> JAX)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Template prototypes Zc (global basis; used to form psi locally)
    # ------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    key, subz = jax.random.split(key)
    Zc = jax.random.normal(subz, (args.clusters, args.emb_dim)) * 0.5

    Zc_norm = Zc / (jnp.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
    W_psi = Zc_norm @ Zc_norm.T  # template compatibility prior

    # template peers: top vocab words by dot(Xw, Zc[c])
    Xw_dot_Z = np.array((Xw @ Zc.T))  # (V,C)
    template_peers: Dict[int, List[int]] = {}
    for c in range(args.clusters):
        idx = np.argsort(-Xw_dot_Z[:, c])[:50]
        template_peers[c] = idx.tolist()

    def peers_for_template(tidx: int) -> str:
        ids = template_peers[int(tidx)][:10]
        return ", ".join([vocab[i] for i in ids])

    # ------------------------------------------------------------
    # Attention projections
    # ------------------------------------------------------------
    key, subq, subk = jax.random.split(key, 3)
    Wq = jax.random.normal(subq, (args.emb_dim, args.head_dim)) / math.sqrt(args.emb_dim)
    Wk = jax.random.normal(subk, (args.emb_dim, args.head_dim)) / math.sqrt(args.emb_dim)

    Kk = min(int(args.topk_attn), int(args.seg_len) - 1)
    NEG_LARGE = -1e9

    @jax.jit
    def compute_psi(word_ids_seg):
        Ew = Xw[word_ids_seg]  # (S,D)
        z_seg = jnp.mean(Ew, axis=0)  # (D,)
        token_scores = Ew @ Zc.T
        ctx_scores = (z_seg @ Zc.T)[None, :]
        scores = token_scores + args.gamma_ctx * ctx_scores
        psi = jax.nn.softmax(scores / args.tau_psi, axis=1)
        return Ew, z_seg, psi

    @jax.jit
    def topk_edges_from_logits(logits):
        """
        logits: (S,S) masked with NEG_LARGE outside candidates/self.
        Return COO edge list (src,dst,w) with exactly S*Kk entries.
        """
        S = logits.shape[0]
        _, topk_idx = jax.lax.top_k(logits, Kk)  # (S,Kk)

        def row_probs(i):
            idxs = topk_idx[i]
            sel = logits[i, idxs]
            sel = sel - jnp.max(sel)
            ex = jnp.exp(sel)
            probs = ex / (jnp.sum(ex) + 1e-12)
            return probs

        probs = jax.vmap(row_probs)(jnp.arange(S))  # (S,Kk)

        src = jnp.repeat(jnp.arange(S), Kk)
        dst = topk_idx.reshape(-1)
        w = probs.reshape(-1)
        return src, dst, w

    @jax.jit
    def j_from_edges(psi, src, dst, w):
        """
        Compute segment-local J_pmi from inferred edges:
          J_mass = (psi[src]*w)^T @ psi[dst]
          J_row = row-normalize
          J_pmi = log(J_row / (pi*pi^T)), standardized
        """
        C = psi.shape[1]
        J_mass = (psi[src] * w[:, None]).T @ psi[dst]  # (C,C)
        J_row = J_mass / (jnp.sum(J_mass, axis=1, keepdims=True) + 1e-12)

        pi_seg = jnp.mean(psi, axis=0)  # (C,)
        base = pi_seg[:, None] * pi_seg[None, :]
        J_pmi = jnp.log((J_row + 1e-12) / (base + 1e-12))
        J_pmi = J_pmi - jnp.mean(J_pmi)
        J_pmi = J_pmi / (jnp.std(J_pmi) + 1e-6)
        return J_pmi

    @jax.jit
    def segment_eventlocal_attention_v2(word_ids_seg, noise_key):
        """
        v2:
          - compute event-local psi
          - build tpl_bias with psi W psi^T + eta*(psi psi^T)
          - compute provisional sparse attention edges WITHOUT dir
          - compute J_pmi from those edges
          - compute final sparse attention edges WITH dir
        Returns:
          src, dst, w_attn (final)
          base_logits, tpl_bias, dir_bias (final)
          psi (event-local)
          pair_mass (template flow from final edges)
        """
        S = word_ids_seg.shape[0]

        Ew, z_seg, psi = compute_psi(word_ids_seg)

        # token embedding anchored with psi@Zc
        eps = args.token_noise * jax.random.normal(noise_key, Ew.shape)
        Eseg = Ew + (psi @ Zc) + eps

        Q = Eseg @ Wq
        K = Eseg @ Wk
        base_logits = (Q @ K.T) / jnp.sqrt(args.head_dim)

        idx = jnp.arange(S)
        dist = jnp.abs(idx[:, None] - idx[None, :])
        local_attn = dist <= args.attn_local_window

        # template bias: compatibility + event-local psi similarity
        tpl_comp = (psi @ W_psi) @ psi.T
        tpl_sim = psi @ psi.T
        tpl_bias = tpl_comp + args.eta_psidot * tpl_sim

        # salience and anti-hub
        idf_tgt = idf_j[word_ids_seg][None, :]
        hub_penalty = -args.lambda_antihub * (1.0 / (idf_tgt + 1e-6))

        # --- provisional logits (no dir) ---
        logits0 = base_logits + args.alpha_template * tpl_bias + args.lambda_idf_target * idf_tgt + hub_penalty
        logits0 = jnp.where(local_attn, logits0, NEG_LARGE)
        logits0 = logits0 - 1e9 * jnp.eye(S, dtype=logits0.dtype)

        src0, dst0, w0 = topk_edges_from_logits(logits0)

        # infer event-local directional coupling from provisional edges
        J_pmi_seg = j_from_edges(psi, src0, dst0, w0)

        # directional bias (soft)
        dir_bias = (psi @ J_pmi_seg) @ psi.T

        # --- final logits (with dir) ---
        logits = logits0 + args.beta_directional * dir_bias
        src, dst, w_attn = topk_edges_from_logits(logits)

        pair_mass = (psi[src] * w_attn[:, None]).T @ psi[dst]  # (C,C)
        return src, dst, w_attn, base_logits, tpl_bias, dir_bias, psi, pair_mass

    # ------------------------------------------------------------
    # Sample segments + compile
    # ------------------------------------------------------------
    starts = [random.randint(0, T - args.seg_len - 1) for _ in range(args.n_segs)]
    print("\nCompiling event-local v2 COO attention (one-time JAX cost)...")
    dummy = tok_ids[starts[0]:starts[0] + args.seg_len].copy()
    dummy[dummy < 0] = 0
    _ = segment_eventlocal_attention_v2(jnp.array(dummy, dtype=jnp.int32), key)[0].block_until_ready()
    print("Compilation finished.\n")

    # ------------------------------------------------------------
    # Run segments, collect edges and flows
    # ------------------------------------------------------------
    all_edges = []
    template_flow = np.zeros((args.clusters, args.clusters), dtype=np.float64)

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start:start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]
        seg_j = jnp.array(seg, dtype=jnp.int32)

        key, nk = jax.random.split(key)
        src, dst, w_attn, baseL, tplB, dirB, psi, pair_mass = segment_eventlocal_attention_v2(seg_j, nk)

        src_np = np.array(src)
        dst_np = np.array(dst)
        w_np = np.array(w_attn)

        base_np = np.array(baseL)
        tpl_np = np.array(tplB)
        dir_np = np.array(dirB)
        psi_np = np.array(psi)

        template_flow += np.array(pair_mass)

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
                "psi_i": psi_np[int(s_idx)].copy(),
                "psi_j": psi_np[int(d_idx)].copy(),
            })

    template_flow /= max(1, args.n_segs)

    # ------------------------------------------------------------
    # Template flow output
    # ------------------------------------------------------------
    print("-" * 80)
    print("TEMPLATE→TEMPLATE ATTENTION FLOWS (aggregated over sampled segments)  [EVENT-LOCAL v2]")
    print("-" * 80)
    pairs = []
    for a in range(args.clusters):
        for b in range(args.clusters):
            if a == b:
                continue
            pairs.append((template_flow[a, b], a, b))
    pairs.sort(reverse=True)
    for val, a, b in pairs[:16]:
        print(f"Template {a:02d} → Template {b:02d}   flow={val:.4f}   Wpsi={float(W_psi[a,b]):.4f}")

    # ------------------------------------------------------------
    # Top edges output
    # ------------------------------------------------------------
    all_edges.sort(key=lambda r: r["score_idf"], reverse=True)
    topN = min(args.top_edges, len(all_edges))
    topM = min(args.top_edges_with_clusters, len(all_edges))

    print("\n" + "-" * 80)
    print(f"TOP {topN} TOKEN→TOKEN LINKS (IDF-weighted; EVENT-LOCAL v2; COO k-sparse)")
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

    # ------------------------------------------------------------
    # Cluster context for top edges
    # ------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"CLUSTER CONTEXT FOR TOP {topM} LINKS (token-local psi + global peers)")
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

    # ------------------------------------------------------------
    # Hierarchy parses (densify only for first 64 tokens)
    # ------------------------------------------------------------
    print("\n" + "-" * 80)
    print("HIERARCHY (binary bracket parses) for sampled segments  [EVENT-LOCAL v2]")
    print("-" * 80)

    for seg_i, start in enumerate(starts, 1):
        seg = tok_ids[start:start + args.seg_len].copy()
        seg[seg < 0] = 0
        seg_words = [vocab[int(w)] for w in seg]
        seg_j = jnp.array(seg, dtype=jnp.int32)

        key, nk = jax.random.split(key)
        src, dst, w_attn, *_ = segment_eventlocal_attention_v2(seg_j, nk)

        src_np = np.array(src)
        dst_np = np.array(dst)
        w_np = np.array(w_attn)

        A = np.zeros((args.seg_len, args.seg_len), dtype=np.float32)
        A[src_np, dst_np] = w_np

        L = min(64, args.seg_len)
        parse = build_parse(A[:L, :L], seg_words[:L], 0, L - 1, max_depth=6)
        print(f"\n[seg {seg_i}] tokens {start}..{start+L-1}")
        print(parse)

    print("\nDone.")


if __name__ == "__main__":
    main()
