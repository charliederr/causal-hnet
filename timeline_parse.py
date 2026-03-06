#!/usr/bin/env python3
"""Neural timeline parser.

Runs the mutual expansion algorithm in the same temporal order as the
neural raster simulation, with interleaved FF→DIS cycles at ~100ms each.
Words arrive at ~400ms intervals → ~4 cycles converge before the next word.

Timeline for a 3-token phrase (e.g. "we should go"):

  T0   t=0         first word arrives
  T1   +400ms      →FF cycle 1 — right-contexts of T0 scored against second input
  T2   +100ms      ←DIS cycle 1 — left-contexts of T1 FF → T0-word class
  T3   +100ms      →FF cycle 2 — right-contexts of T2 class → refined T1-word subs
  T4   +100ms      ←DIS cycle 2 — converged T0-word class (~400ms elapsed)
  T5   +100ms      →FF — third input arrives; right-contexts of T4 → T2-word subs
  T6   +100ms      ←DIS *C — C's left-context disinhibits left antecedent
                   = competition: *C → A'+B' (AB|C) or *(B'+C) → A' (A|BC)
                   Three-element asymmetry makes this step competitive.
  T7   +100ms      →FF * — right-contexts of competition winners → next-word predictions
                   FF_AB: continuation of AB|C parse path
                   FF_A:  continuation of A|BC parse path

Convergence: each cycle filters both sides by the other's context.
Competition fires when a third element creates an asymmetric split via DIS.

Output: same NeuralRasterGUI display as neural_demo.py.

Usage:
    python3 timeline_parse.py
    python3 timeline_parse.py "she might leave"
    python3 timeline_parse.py "we should go" --top-n 15
    python3 timeline_parse.py "we should go" --save-png /tmp/timeline.png
    python3 timeline_parse.py "i found my hat"
    python3 timeline_parse.py "i found my hat" --save-png /tmp/timeline_4w.png
"""

import sys
import os
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ContextPattern must be importable before pickle.load
from bidir_simple import UnitCatalog, ContextPattern  # noqa: F401
from analyze_phrase import get_unit_with_fallback


# ── Core computation helpers ──────────────────────────────────────────

def get_ctx(word, catalog, corpus_index=None):
    """Return (left_words, right_words) list pair for a unit."""
    pat = get_unit_with_fallback(word, catalog, corpus_index)
    if pat:
        return list(pat.left_words.keys()), list(pat.right_words.keys())
    return [], []


def score_subs(catalog, target, candidates, max_n, scoring='containment'):
    """Score candidates as substitutes for target; return top max_n."""
    if not candidates:
        return []
    results = catalog.gpu_contextual_candidates(
        target=target,
        candidates=candidates,
        max_results=max_n * 3,
        scoring=scoring,
        trace=False,
    )
    return [(t, s) for t, s in results if t != target][:max_n]


def consensus_ctx(subs, catalog, corpus_index, side, min_count=2):
    """Collect left- or right-context words shared by >= min_count subs."""
    counts = {}
    for word, _ in subs:
        pat = get_unit_with_fallback(word, catalog, corpus_index)
        if pat:
            pool = pat.right_words if side == 'right' else pat.left_words
            for w in pool.keys():
                counts[w] = counts.get(w, 0) + 1
    return [w for w, c in counts.items() if c >= min_count]


def scored_consensus_ctx(subs, catalog, corpus_index, side, min_count=2, max_n=200):
    """Like consensus_ctx but returns (word, normalized_count) pairs, sorted by count."""
    counts = {}
    for word, _ in subs:
        pat = get_unit_with_fallback(word, catalog, corpus_index)
        if pat:
            pool = pat.right_words if side == 'right' else pat.left_words
            for w in pool.keys():
                counts[w] = counts.get(w, 0) + 1
    filtered = [(w, c) for w, c in counts.items() if c >= min_count]
    if not filtered:
        return []
    max_c = max(c for _, c in filtered)
    return [(w, c / max_c) for w, c in sorted(filtered, key=lambda x: -x[1])][:max_n]


def add_bigram(catalog, bigram, left_ctx, right_ctx):
    """Register a bigram phrase in the GPU index."""
    catalog.add_unit_to_gpu_index(
        bigram,
        Counter({w: 1 for w in left_ctx}),
        Counter({w: 1 for w in right_ctx}),
    )


# ── Timeline computation ──────────────────────────────────────────────

def compute_timeline(phrase, catalog, corpus_index=None, top_n=10, max_n=200):
    """
    Compute all eight time bins for a 3-token phrase.

    Returns:
        results: list of (time_bin, population, word, score, rank)
        tokens:  first three lowercase tokens
    """
    tokens = phrase.lower().split()
    if len(tokens) < 3:
        raise ValueError(f'Need at least 3 tokens (got {len(tokens)}): {tokens}')

    w0, w1, w2 = tokens[:3]

    # Context patterns for each word from catalog
    left_0, right_0 = get_ctx(w0, catalog, corpus_index)
    left_1, right_1 = get_ctx(w1, catalog, corpus_index)
    left_2, right_2 = get_ctx(w2, catalog, corpus_index)

    results = []

    def emit(tb, pop, items):
        for rank, (word, score) in enumerate(items[:top_n]):
            results.append((tb, pop, word, score, rank))

    # ── T0: first word ────────────────────────────────────────────────
    emit(0, 'INPUT', [(w0, 1.0)])

    # ── T1: →FF cycle 1 ──────────────────────────────────────────────
    # Right-contexts of w0 scored against w1 → w1-class substitutes
    ff_1 = score_subs(catalog, w1, right_0, max_n)
    emit(1, 'INPUT', [(w1, 1.0)])
    emit(1, 'FF',    ff_1)

    # ── T2: ←DIS cycle 1 ─────────────────────────────────────────────
    # Left-contexts of T1 FF burst → w0-class substitutes
    ff1_left_ctx = consensus_ctx(ff_1[:top_n], catalog, corpus_index,
                                 side='left', min_count=2)
    dis_2 = score_subs(catalog, w0, ff1_left_ctx, max_n)
    emit(2, 'DIS', dis_2)

    # ── T3: →FF cycle 2 (refined) ─────────────────────────────────────
    # Right-contexts of T2 DIS burst → refined w1-class substitutes
    # Only candidates appearing in right-context of multiple T2 pronouns survive.
    dis2_right_ctx = consensus_ctx(dis_2[:top_n], catalog, corpus_index,
                                   side='right', min_count=2)
    ff_3 = score_subs(catalog, w1, dis2_right_ctx, max_n)
    emit(3, 'FF', ff_3)

    # ── T4: ←DIS cycle 2 (converged) ─────────────────────────────────
    # Left-contexts of T3 refined FF → converged w0-class substitutes
    ff3_left_ctx = consensus_ctx(ff_3[:top_n], catalog, corpus_index,
                                 side='left', min_count=2)
    dis_4 = score_subs(catalog, w0, ff3_left_ctx, max_n)
    emit(4, 'DIS', dis_4)

    # ── T5: →FF "go" ─────────────────────────────────────────────────
    # w2 arrives; right-contexts of converged T3 FF (w1-class) scored against w2
    # The T3 modals' right-contexts give what follows modal verbs → infinitives
    ff3_right_ctx = consensus_ctx(ff_3[:top_n], catalog, corpus_index,
                                  side='right', min_count=2)
    ff_5 = score_subs(catalog, w2, ff3_right_ctx, max_n)
    emit(5, 'INPUT', [(w2, 1.0)])
    emit(5, 'FF',    ff_5)

    # ── T6: ←DIS *C — competition via three-element asymmetry ────────
    # C's left-context disinhibits the left antecedent.
    # The asymmetry of three elements forces a choice of parse structure:
    #
    # AB|C parse: RIGHT = C alone ("go").  LEFT = entire AB unit ("we should").
    #   DIS from C reveals 2-word (subject, modal) bigram substitutes for "we should".
    #   Step 1: score modals from C's left-context as substitutes for w1 ("should").
    #   Step 2: collect consensus subjects from those modals' left-contexts.
    #   Step 3: score subjects as substitutes for w0 ("we").
    #   Step 4: emit (subject modal) pairs — 2-word replacements for the AB unit.
    #
    # A|BC parse: RIGHT = BC unit ("should go").  LEFT = just A ("we").
    #   DIS from BC (registered with left_1 as its left-context) reveals
    #   single-word substitutes for w0 ("we").

    add_bigram(catalog, f'{w0} {w1}', left_0, right_1)
    add_bigram(catalog, f'{w1} {w2}', left_1, right_2)

    # AB|C — two-step DIS cascade from C's left-context
    comp_AB_modals = score_subs(catalog, w1, left_2, max_n)
    comp_AB_subj_pool = consensus_ctx(comp_AB_modals[:top_n], catalog, corpus_index,
                                      side='left', min_count=2)
    comp_AB_subjs = score_subs(catalog, w0, comp_AB_subj_pool, max_n)
    # Cross-product of top subjects × top modals, scored by product
    comp_AB_pairs = [
        (f'{subj} {modal}', s_sc * m_sc)
        for subj, s_sc in comp_AB_subjs[:top_n]
        for modal, m_sc in comp_AB_modals[:top_n]
    ]
    comp_AB_pairs.sort(key=lambda x: -x[1])
    comp_AB = comp_AB_pairs[:max_n]

    # A|BC — single-word DIS from BC's left-context (= left_1, "should"'s left-context)
    comp_A = score_subs(catalog, w0, left_1, max_n)

    emit(6, 'COMP_AB', comp_AB)
    emit(6, 'COMP_A',  comp_A)

    # ── T7: →FF continuation ──────────────────────────────────────────
    # AB|C parse: C is a single word → FF predicts single-word substitutes for w2.
    #   Right-contexts of the AB|C modal winners → score as substitutes for w2.
    #
    # A|BC parse: BC is a 2-word unit → FF predicts bigram (modal, verb) substitutes
    #   for "should go" as a whole. Two-step cascade from the A|BC subject winners:
    #   Step 1: right-contexts of subjects → modal pool → score as substitutes for w1.
    #   Step 2: right-contexts of those modals → verb pool → score as substitutes for w2.
    #   Emit (modal verb) pairs — 2-word replacements for the BC unit.

    # FF_AB: single-word C-element substitutes for w2 ("go")
    ff_ab_verb_pool = consensus_ctx(comp_AB_modals[:top_n], catalog, corpus_index,
                                    side='right', min_count=2)
    ff_7_AB = score_subs(catalog, w2, ff_ab_verb_pool, max_n)

    # FF_A: bigram BC-element substitutes for "should go"
    ff_a_modal_pool = consensus_ctx(comp_A[:top_n], catalog, corpus_index,
                                    side='right', min_count=2)
    ff_a_modals = score_subs(catalog, w1, ff_a_modal_pool, max_n)
    ff_a_verb_pool = consensus_ctx(ff_a_modals[:top_n], catalog, corpus_index,
                                   side='right', min_count=2)
    ff_a_verbs = score_subs(catalog, w2, ff_a_verb_pool, max_n)
    ff_7_A_pairs = [
        (f'{modal} {verb}', m_sc * v_sc)
        for modal, m_sc in ff_a_modals[:top_n]
        for verb, v_sc  in ff_a_verbs[:top_n]
    ]
    ff_7_A_pairs.sort(key=lambda x: -x[1])
    ff_7_A = ff_7_A_pairs[:max_n]
    emit(7, 'FF_AB', ff_7_AB)
    emit(7, 'FF_A',  ff_7_A)

    return results, tokens[:3]


def compute_timeline_4w(phrase, catalog, corpus_index=None, top_n=10, max_n=200):
    """
    Compute all eleven time bins for a 4-token phrase.

    T0–T7: same 3-word processing for (w0, w1, w2).
    T8–T10: 4-word extension when D = w3 arrives.

      T8   +100ms   →FF D arrives — noun-class scored from det right-contexts
      T9   +100ms   ←DIS *D — 4-word competition:
                     COMP_L3 (ABC|D): 3-step cascade from left_3 → (subj, verb, det) trigrams
                     COMP_L2 (AB|CD): reactivation of T6 COMP_AB (left_2 context)
                     COMP_L1 (A|BCD): reactivation of T6 COMP_A  (left_1 context)
      T10  +100ms   →FF ** — 4-word continuations:
                     FF_R1 (ABC|D): 1-word D-class
                     FF_R2 (AB|CD): 2-word (det, noun) CD-class
                     FF_R3 (A|BCD): 3-word (verb, det, noun) BCD-class

    Returns:
        results: list of (time_bin, population, word, score, rank)
        tokens:  first four lowercase tokens
    """
    tokens = phrase.lower().split()
    if len(tokens) < 4:
        raise ValueError(f'Need at least 4 tokens (got {len(tokens)}): {tokens}')

    w0, w1, w2, w3 = tokens[:4]

    left_0, right_0 = get_ctx(w0, catalog, corpus_index)
    left_1, right_1 = get_ctx(w1, catalog, corpus_index)
    left_2, right_2 = get_ctx(w2, catalog, corpus_index)
    left_3, right_3 = get_ctx(w3, catalog, corpus_index)

    results = []

    def emit(tb, pop, items):
        for rank, (word, score) in enumerate(items[:top_n]):
            results.append((tb, pop, word, score, rank))

    # ── T0–T7: 3-word processing for (w0, w1, w2) ────────────────────
    emit(0, 'INPUT', [(w0, 1.0)])

    ff_1 = score_subs(catalog, w1, right_0, max_n)
    emit(1, 'INPUT', [(w1, 1.0)])
    emit(1, 'FF',    ff_1)

    ff1_left_ctx = consensus_ctx(ff_1[:top_n], catalog, corpus_index,
                                 side='left', min_count=2)
    dis_2 = score_subs(catalog, w0, ff1_left_ctx, max_n)
    emit(2, 'DIS', dis_2)

    dis2_right_ctx = consensus_ctx(dis_2[:top_n], catalog, corpus_index,
                                   side='right', min_count=2)
    ff_3 = score_subs(catalog, w1, dis2_right_ctx, max_n)
    emit(3, 'FF', ff_3)

    ff3_left_ctx = consensus_ctx(ff_3[:top_n], catalog, corpus_index,
                                 side='left', min_count=2)
    dis_4 = score_subs(catalog, w0, ff3_left_ctx, max_n)
    emit(4, 'DIS', dis_4)

    ff3_right_ctx = consensus_ctx(ff_3[:top_n], catalog, corpus_index,
                                  side='right', min_count=2)
    ff_5 = score_subs(catalog, w2, ff3_right_ctx, max_n)
    emit(5, 'INPUT', [(w2, 1.0)])
    emit(5, 'FF',    ff_5)

    add_bigram(catalog, f'{w0} {w1}', left_0, right_1)
    add_bigram(catalog, f'{w1} {w2}', left_1, right_2)

    comp_AB_modals = score_subs(catalog, w1, left_2, max_n)
    comp_AB_subj_pool = consensus_ctx(comp_AB_modals[:top_n], catalog, corpus_index,
                                      side='left', min_count=2)
    comp_AB_subjs = score_subs(catalog, w0, comp_AB_subj_pool, max_n)
    comp_AB_pairs = [
        (f'{subj} {modal}', s_sc * m_sc)
        for subj, s_sc in comp_AB_subjs[:top_n]
        for modal, m_sc in comp_AB_modals[:top_n]
    ]
    comp_AB_pairs.sort(key=lambda x: -x[1])
    comp_AB = comp_AB_pairs[:max_n]

    comp_A = score_subs(catalog, w0, left_1, max_n)
    emit(6, 'COMP_AB', comp_AB)
    emit(6, 'COMP_A',  comp_A)

    ff_ab_verb_pool = consensus_ctx(comp_AB_modals[:top_n], catalog, corpus_index,
                                    side='right', min_count=2)
    ff_7_AB = score_subs(catalog, w2, ff_ab_verb_pool, max_n)

    ff_a_modal_pool = consensus_ctx(comp_A[:top_n], catalog, corpus_index,
                                    side='right', min_count=2)
    ff_a_modals = score_subs(catalog, w1, ff_a_modal_pool, max_n)
    ff_a_verb_pool = consensus_ctx(ff_a_modals[:top_n], catalog, corpus_index,
                                   side='right', min_count=2)
    ff_a_verbs = score_subs(catalog, w2, ff_a_verb_pool, max_n)
    ff_7_A = sorted([
        (f'{modal} {verb}', m_sc * v_sc)
        for modal, m_sc in ff_a_modals[:top_n]
        for verb, v_sc  in ff_a_verbs[:top_n]
    ], key=lambda x: -x[1])[:max_n]
    emit(7, 'FF_AB', ff_7_AB)
    emit(7, 'FF_A',  ff_7_A)

    # ── T8: →FF + D = w3 arrives ──────────────────────────────────────
    # Right-contexts of T7 C-element substitutes (ff_7_AB = det/prep class)
    # scored against w3 → D-class noun substitutes
    ff8_pool = consensus_ctx(ff_7_AB[:top_n], catalog, corpus_index,
                             side='right', min_count=2)
    ff_8 = score_subs(catalog, w3, ff8_pool, max_n)
    emit(8, 'INPUT', [(w3, 1.0)])
    emit(8, 'FF',    ff_8)

    # ── T9: ←DIS *D — 4-word competition ─────────────────────────────
    # Register CD bigram and ABC/BCD multi-word units
    add_bigram(catalog, f'{w2} {w3}',       left_2, right_3)   # CD
    add_bigram(catalog, f'{w0} {w1} {w2}',  left_0, right_2)   # ABC
    add_bigram(catalog, f'{w1} {w2} {w3}',  left_1, right_3)   # BCD

    # COMP_L3 (ABC|D): 3-step DIS cascade from D's left-context
    # left_3 = left-contexts of w3 ("hat") = det/possessive class
    comp_l3_dets  = score_subs(catalog, w2, left_3, max_n)          # det-subs for w2
    comp_l3_vpool = consensus_ctx(comp_l3_dets[:top_n], catalog, corpus_index,
                                  side='left', min_count=2)
    comp_l3_verbs = score_subs(catalog, w1, comp_l3_vpool, max_n)   # verb-subs for w1
    comp_l3_spool = consensus_ctx(comp_l3_verbs[:top_n], catalog, corpus_index,
                                  side='left', min_count=2)
    comp_l3_subjs = score_subs(catalog, w0, comp_l3_spool, max_n)   # subj-subs for w0
    comp_L3 = sorted([
        (f'{s} {v} {d}', s_sc * v_sc * d_sc)
        for s, s_sc in comp_l3_subjs[:top_n]
        for v, v_sc in comp_l3_verbs[:top_n]
        for d, d_sc in comp_l3_dets[:top_n]
    ], key=lambda x: -x[1])[:max_n]

    # COMP_L2 (AB|CD): reactivation — CD's left-context = left_2, same as T6 COMP_AB
    comp_L2 = comp_AB  # identical computation (left_2 context unchanged)

    # COMP_L1 (A|BCD): reactivation — BCD's left-context = left_1, same as T6 COMP_A
    comp_L1 = comp_A   # identical computation (left_1 context unchanged)

    emit(9, 'COMP_L3', comp_L3)
    emit(9, 'COMP_L2', comp_L2)
    emit(9, 'COMP_L1', comp_L1)

    # ── T10: →FF ** — 4-word continuations ────────────────────────────

    # FF_R1 (ABC|D): single-word D-class — right-contexts of det winners → noun subs
    ff_r1_pool = consensus_ctx(comp_l3_dets[:top_n], catalog, corpus_index,
                               side='right', min_count=2)
    ff_R1 = score_subs(catalog, w3, ff_r1_pool, max_n)

    # FF_R2 (AB|CD): 2-word (det, noun) CD-class — 2-step from verb right-contexts
    ff_r2_dpool = consensus_ctx(comp_AB_modals[:top_n], catalog, corpus_index,
                                side='right', min_count=2)
    ff_r2_dets   = score_subs(catalog, w2, ff_r2_dpool, max_n)
    ff_r2_npool  = consensus_ctx(ff_r2_dets[:top_n], catalog, corpus_index,
                                 side='right', min_count=2)
    ff_r2_nouns  = score_subs(catalog, w3, ff_r2_npool, max_n)
    ff_R2 = sorted([
        (f'{d} {n}', d_sc * n_sc)
        for d, d_sc in ff_r2_dets[:top_n]
        for n, n_sc in ff_r2_nouns[:top_n]
    ], key=lambda x: -x[1])[:max_n]

    # FF_R3 (A|BCD): 3-word (verb, det, noun) BCD-class — 3-step from subject right-contexts
    ff_r3_vpool  = consensus_ctx(comp_A[:top_n], catalog, corpus_index,
                                 side='right', min_count=2)
    ff_r3_verbs  = score_subs(catalog, w1, ff_r3_vpool, max_n)
    ff_r3_dpool  = consensus_ctx(ff_r3_verbs[:top_n], catalog, corpus_index,
                                 side='right', min_count=2)
    ff_r3_dets   = score_subs(catalog, w2, ff_r3_dpool, max_n)
    ff_r3_npool  = consensus_ctx(ff_r3_dets[:top_n], catalog, corpus_index,
                                 side='right', min_count=2)
    ff_r3_nouns  = score_subs(catalog, w3, ff_r3_npool, max_n)
    ff_R3 = sorted([
        (f'{v} {d} {n}', v_sc * d_sc * n_sc)
        for v, v_sc in ff_r3_verbs[:top_n]
        for d, d_sc in ff_r3_dets[:top_n]
        for n, n_sc in ff_r3_nouns[:top_n]
    ], key=lambda x: -x[1])[:max_n]

    emit(10, 'FF_R1', ff_R1)
    emit(10, 'FF_R2', ff_R2)
    emit(10, 'FF_R3', ff_R3)

    return results, tokens[:4]


# ── Build TemporalSpikeEvents ─────────────────────────────────────────

def build_events(results):
    from neural_raster import TemporalSpikeEvent
    return [
        TemporalSpikeEvent(
            time_bin=tb, word=word, score=score,
            population=pop, rank=rank,
        )
        for tb, pop, word, score, rank in results
    ]


# ── Entry point ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phrase', nargs='?', default='we should go',
                    help='Phrase to analyze (default: "we should go")')
    ap.add_argument('--top-n', type=int, default=10,
                    help='Items to show per population per bin (default: 10)')
    ap.add_argument('--max-n', type=int, default=200,
                    help='Candidates computed before trimming (default: 200)')
    ap.add_argument('--scoring', default='containment',
                    choices=['containment', 'cosine', 'ic_cosine'])
    ap.add_argument('--min-freq', type=int, default=10,
                    help='GPU index minimum frequency (default: 10)')
    ap.add_argument('--catalog', default=None,
                    help='Path to unit_catalog.pkl')
    ap.add_argument('--save-png', default=None,
                    help='Save figure to PNG instead of showing interactively')
    args = ap.parse_args()

    if args.save_png:
        import matplotlib
        matplotlib.use('Agg')

    catalog_path = args.catalog or os.path.join(
        os.path.dirname(__file__), 'unit_catalog.pkl')

    print('Loading catalog...')
    catalog = UnitCatalog()
    catalog.load(catalog_path)
    catalog.build_gpu_index(min_freq=args.min_freq)
    print(f'Catalog loaded ({len(catalog.units)} units).\n')

    n_tokens = len(args.phrase.split())
    print(f'Computing timeline for: "{args.phrase}" ({n_tokens} tokens)')

    from neural_raster import NeuralRasterGUI

    if n_tokens >= 4:
        from neural_raster import (
            POPULATIONS_4W, POP_LABELS_4W, POP_COLORS_4W, POP_TOP_N_4W,
            N_TIME_BINS_4W, TIME_BIN_LABELS_4W, TIME_BIN_COLORS_4W,
        )
        results, tokens = compute_timeline_4w(
            args.phrase, catalog,
            top_n=args.top_n,
            max_n=args.max_n,
        )
        events = build_events(results)
        title  = f'Neural Timeline: "{args.phrase}"'
        gui = NeuralRasterGUI(events, title=title,
                              populations=POPULATIONS_4W,
                              pop_labels=POP_LABELS_4W,
                              pop_colors=POP_COLORS_4W,
                              pop_top_n=POP_TOP_N_4W,
                              n_time_bins=N_TIME_BINS_4W,
                              time_bin_labels=TIME_BIN_LABELS_4W,
                              time_bin_colors=TIME_BIN_COLORS_4W)
    else:
        results, tokens = compute_timeline(
            args.phrase, catalog,
            top_n=args.top_n,
            max_n=args.max_n,
        )
        events = build_events(results)
        title  = f'Neural Timeline: "{args.phrase}"'
        gui = NeuralRasterGUI(events, title=title)

    if args.save_png:
        gui.fig.savefig(args.save_png, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.save_png}')
    else:
        gui.show()


if __name__ == '__main__':
    main()
