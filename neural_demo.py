#!/usr/bin/env python3
"""Neural temporal raster demo for "we should go".

Maps the expansion algorithm onto interleaved FF→DIS cycles.
Each cycle takes ~100ms; words arrive at ~400ms intervals → ~4 cycles
converge before the next word, fully filtering each element by consensus
of both right and left contexts.

  T0  (t=0)         "we" arrives as input
  T1  (t+400ms)     →FF cycle 1: right-contexts of "we" scored against "should"
                    → modal verbs that fit "we ___"  (A* + B)
  T2  (t+100ms)     ←DIS cycle 1: left-contexts of T1 modals → "we"-like words
                    → pronouns that precede modals  (*(A*+B))
  T3  (t+100ms)     →FF cycle 2: right-contexts of T2 pronouns scored against "should"
                    → refined modal set, higher consensus
  T4  (t+100ms)     ←DIS cycle 2: left-contexts of T3 modals → converged "we"-like
                    → contracted pronoun forms dominate
                    [~400ms elapsed since T1; representation converged]
  T5  (t+100ms)     →FF: "go" arrives, right-contexts of T4-converged context
                    scored against "go"  ((A*+B)*+C)
  T6  (t+100ms)     ←DIS: left-contexts of T5 verbs → filtered "should"-like words
                    → modals consistent with both "we"-space and "go"-space
  T7  (t+~300ms)    Competition: three-element asymmetry forces parse selection
                      COMP_AB  →  AB|C parse: "we should" | "go"
                      COMP_A   →  A|BC parse: "we" | "should go"

Convergence property: T4 pronouns are filtered by being consistent with
T3 modals (which came from T2 pronoun right-contexts, which came from T1
modals filtered by T0-word context). Each cycle tightens both sides.

Usage:
    python3 neural_demo.py
    python3 neural_demo.py --save-png /tmp/neural_raster.png
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_raster import TemporalSpikeEvent


def build_temporal_events():
    """Construct TemporalSpikeEvents for 'we should go'."""
    events = []

    def add(time_bin, population, words_scores):
        for rank, (word, score) in enumerate(words_scores):
            events.append(TemporalSpikeEvent(
                time_bin=time_bin, word=word, score=score,
                population=population, rank=rank))

    # ── T0: "we" arrives ────────────────────────────────────────────
    add(0, 'INPUT', [('we', 1.0)])

    # ── T1: →FF cycle 1 ─────────────────────────────────────────────
    # Right-contexts of "we" scored against "should" → modal verbs (A* + B)
    add(1, 'INPUT', [('should', 1.0)])
    add(1, 'FF', [
        ('could',    0.92), ('must',     0.85), ('might',    0.83), ('may',      0.80),
        ("won't",    0.65), ('will',     0.60), ('can',      0.55), ("wouldn't", 0.48),
    ])

    # ── T2: ←DIS cycle 1 ────────────────────────────────────────────
    # Left-contexts of T1 modal burst → "we"-like words (*(A*+B))
    # Pronouns that precede modals in corpus.
    add(2, 'DIS', [
        ('they',    0.90), ('you',    0.78), ('i',      0.75), ("you'll", 0.60),
        ("they'd",  0.58), ("she'd",  0.45), ("he'll",  0.42), ("we'd",   0.38),
    ])

    # ── T3: →FF cycle 2 (refined) ───────────────────────────────────
    # Right-contexts of T2 pronouns scored against "should" → refined modals.
    # Only modals appearing in the right context of multiple pronouns survive.
    # "won't"/"wouldn't" drop out (less consensus in pronoun right-contexts).
    add(3, 'FF', [
        ('could',  0.94), ('must',  0.90), ('might', 0.87), ('may',  0.83),
        ('can',    0.76), ('will',  0.70),
    ])

    # ── T4: ←DIS cycle 2 (converged) ────────────────────────────────
    # Left-contexts of T3 refined modals → converged "we"-like words.
    # Contracted pronoun+modal forms emerge as the stable attractor.
    add(4, 'DIS', [
        ("they'd", 0.88), ("we'd",  0.82), ("i'd",   0.78), ('they',  0.72),
        ("you'd",  0.65), ('you',   0.58), ('i',      0.52), ("she'd", 0.45),
    ])

    # ── T5: →FF "go" ────────────────────────────────────────────────
    # "go" arrives; right-contexts of converged modal set scored against "go"
    # → infinitive verbs that follow "could/must/might/..." ((A*+B)*+C)
    add(5, 'INPUT', [('go', 1.0)])
    add(5, 'FF', [
        ('speak',  0.88), ('begin',  0.82), ('try',    0.78), ('come',   0.75),
        ('learn',  0.70), ('move',   0.62), ('fly',    0.50), ('play',   0.42),
    ])

    # ── T6: ←DIS *C — competition via three-element asymmetry ───────
    # AB|C parse: RIGHT = C ("go") alone.  LEFT = entire AB unit ("we should").
    #   DIS from C → 2-word (subject modal) bigrams replacing "we should":
    #   "they'll"="they will", "we'll"="we will", "i'll"="i will" etc.
    add(6, 'COMP_AB', [
        ("they'll", 0.88), ("we'll",  0.82), ("i'll",   0.78), ("they'd", 0.72),
        ("you'll",  0.65), ("she'd",  0.55), ("he'll",  0.48), ("it'll",  0.40),
    ])

    # A|BC parse: RIGHT = BC unit ("should go").  LEFT = just A ("we").
    #   DIS from BC → single-word subject substitutes for "we":
    add(6, 'COMP_A', [
        ('they',    0.88), ('you',    0.75), ('i',      0.72), ("they'd",  0.58),
        ("he'd",    0.52), ("she'd",  0.45), ('people', 0.35), ('someone', 0.28),
    ])

    # ── T7: →FF continuation ─────────────────────────────────────────
    # AB|C parse: C is single word → single-word verb substitutes for "go"
    add(7, 'FF_AB', [
        ('speak',  0.88), ('begin',  0.82), ('try',    0.78), ('come',   0.75),
        ('learn',  0.70), ('move',   0.62), ('fly',    0.50), ('play',   0.42),
    ])
    # A|BC parse: BC is 2-word unit → bigram (modal verb) substitutes for "should go"
    add(7, 'FF_A', [
        ('must speak',  0.88), ('could try',   0.82), ('must learn',  0.78),
        ('could learn', 0.74), ('must try',    0.70), ('might come',  0.64),
        ("won't leave", 0.50), ('can start',   0.42),
    ])

    return events


def build_temporal_events_4w():
    """Construct TemporalSpikeEvents for 'I found my hat' (4-word timeline)."""
    events = []

    def add(time_bin, population, words_scores):
        for rank, (word, score) in enumerate(words_scores):
            events.append(TemporalSpikeEvent(
                time_bin=time_bin, word=word, score=score,
                population=population, rank=rank))

    # ── T0-T7: 3-word processing of "I found my" ────────────────────

    # T0: "I" arrives
    add(0, 'INPUT', [('i', 1.0)])

    # T1: →FF cycle 1 — right-contexts of "I" scored against "found"
    add(1, 'INPUT', [('found', 1.0)])
    add(1, 'FF', [
        ('lost',  0.90), ('left',    0.85), ('kept',  0.80), ('saw',     0.75),
        ('wore',  0.65), ('took',    0.60), ('held',  0.55), ('brought', 0.48),
    ])

    # T2: ←DIS cycle 1 — left-contexts of T1 verb burst → "I"-class subjects
    add(2, 'DIS', [
        ('she',     0.88), ('he',     0.82), ('they',    0.75), ('we',      0.65),
        ('you',     0.58), ('someone', 0.52), ('nobody', 0.45), ('everyone', 0.38),
    ])

    # T3: →FF cycle 2 — right-contexts of T2 subjects → refined verb class
    add(3, 'FF', [
        ('lost',  0.92), ('left',  0.88), ('kept', 0.85), ('found',  0.80),
        ('took',  0.74), ('saw',   0.68),
    ])

    # T4: ←DIS cycle 2 — left-contexts of T3 verbs → converged subject class
    add(4, 'DIS', [
        ("she'd", 0.90), ("he'd",   0.85), ("they'd", 0.78), ("i'd",   0.72),
        ("we'd",  0.65), ("you'd",  0.58), ('she',    0.48), ('he',    0.40),
    ])

    # T5: →FF — "my" arrives; right-contexts of T3 verbs scored against "my"
    add(5, 'INPUT', [('my', 1.0)])
    add(5, 'FF', [
        ('your', 0.88), ('his', 0.85), ('her',   0.80), ('their', 0.75),
        ('the',  0.68), ('our', 0.58), ('a',      0.50), ('that',  0.42),
    ])

    # T6: ←DIS *my — 3-word competition via "my"'s left-context
    # AB|C: "my"'s left-context drives (subject, verb) bigrams replacing "I found"
    add(6, 'COMP_AB', [
        ('she lost',   0.88), ('he kept',    0.82), ('they left',  0.78),
        ("she'd lost", 0.72), ('i left',     0.65), ('we found',   0.58),
        ('he lost',    0.52), ("they'd kept", 0.45),
    ])
    # A|BC: "found my" left-context drives single-word subject substitutes for "I"
    add(6, 'COMP_A', [
        ('she', 0.90), ('he',      0.85), ('they',     0.78), ('we',       0.70),
        ('you', 0.62), ('someone', 0.52), ('everyone', 0.42), ('nobody',   0.35),
    ])

    # T7: →FF * — 3-word parse-path continuations
    # AB|C: C = "my"-class (determiners/possessives)
    add(7, 'FF_AB', [
        ('your', 0.88), ('his', 0.85), ('her',   0.80), ('their', 0.75),
        ('the',  0.68), ('our', 0.58), ('a',      0.50), ('that',  0.42),
    ])
    # A|BC: BC = "found my"-class — (verb, determiner) bigrams
    add(7, 'FF_A', [
        ('lost his',   0.85), ('kept her',  0.80), ('left your', 0.75),
        ('took their', 0.70), ('found my',  0.65), ('wore his',  0.58),
        ('brought her', 0.52), ('saw the',  0.45),
    ])

    # ── T8-T10: 4-word extension — "hat" arrives ────────────────────

    # T8: →FF + "hat" arrives; right-contexts of T7 FF_AB (determiners) → noun class
    add(8, 'INPUT', [('hat', 1.0)])
    add(8, 'FF', [
        ('wallet', 0.88), ('bag',     0.85), ('phone',   0.80), ('keys',    0.75),
        ('glasses', 0.70), ('book',   0.62), ('coat',    0.55), ('ring',    0.48),
    ])

    # T9: ←DIS *hat — 4-word competition via "hat"'s left-context
    # ABC|D: 3-step cascade from "hat"'s left-context → (subject, verb, det) trigrams
    add(9, 'COMP_L3', [
        ('she lost her',    0.85), ('he kept his',    0.80), ('they found their', 0.75),
        ('she left her',    0.70), ('he lost his',    0.65), ('we kept our',      0.58),
        ('she took her',    0.52), ('he brought his', 0.45),
    ])
    # AB|CD: reactivation of T6 COMP_AB in 4-word context — (subject, verb) bigrams
    add(9, 'COMP_L2', [
        ('she lost',   0.86), ('he kept',    0.80), ('they left',  0.76),
        ("she'd lost", 0.70), ('i left',     0.63), ('we found',   0.56),
        ('he lost',    0.50), ("they'd kept", 0.43),
    ])
    # A|BCD: reactivation of T6 COMP_A in 4-word context — single-word subjects
    add(9, 'COMP_L1', [
        ('she', 0.88), ('he',      0.83), ('they',     0.76), ('we',       0.68),
        ('you', 0.60), ('someone', 0.50), ('everyone', 0.40), ('nobody',   0.33),
    ])

    # T10: →FF ** — 4-word parse-path continuations
    # ABC|D: D = "hat"-class (1-word nouns)
    add(10, 'FF_R1', [
        ('wallet', 0.88), ('bag',     0.85), ('phone',   0.80), ('keys',    0.75),
        ('glasses', 0.70), ('book',   0.62), ('coat',    0.55), ('ring',    0.48),
    ])
    # AB|CD: CD = "my hat"-class — (determiner, noun) bigrams
    add(10, 'FF_R2', [
        ('her bag',    0.88), ('his wallet',  0.85), ('their keys',  0.80),
        ('her phone',  0.75), ('his glasses', 0.70), ('the book',    0.62),
        ('our coat',   0.55), ('his ring',    0.48),
    ])
    # A|BCD: BCD = "found my hat"-class — (verb, det, noun) trigrams
    add(10, 'FF_R3', [
        ('lost his wallet',  0.85), ('kept her bag',    0.80), ('left their keys', 0.75),
        ('took his phone',   0.70), ('lost her glasses', 0.65), ('brought our book', 0.58),
        ('wore his coat',    0.52), ('found his ring',  0.45),
    ])

    return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-png', default=None)
    parser.add_argument('--phrase', choices=['3w', '4w'], default='3w',
                        help='3w = "we should go", 4w = "I found my hat"')
    args = parser.parse_args()

    if args.save_png:
        import matplotlib
        matplotlib.use('Agg')

    from neural_raster import (
        NeuralRasterGUI,
        POPULATIONS_4W, POP_LABELS_4W, POP_COLORS_4W, POP_TOP_N_4W,
        N_TIME_BINS_4W, TIME_BIN_LABELS_4W, TIME_BIN_COLORS_4W,
    )

    if args.phrase == '4w':
        events = build_temporal_events_4w()
        title  = 'Neural Temporal Raster: "I found my hat"'
        gui = NeuralRasterGUI(events, title=title,
                              populations=POPULATIONS_4W,
                              pop_labels=POP_LABELS_4W,
                              pop_colors=POP_COLORS_4W,
                              pop_top_n=POP_TOP_N_4W,
                              n_time_bins=N_TIME_BINS_4W,
                              time_bin_labels=TIME_BIN_LABELS_4W,
                              time_bin_colors=TIME_BIN_COLORS_4W)
    else:
        events = build_temporal_events()
        title  = 'Neural Temporal Raster: "we should go"'
        gui = NeuralRasterGUI(events, title=title)

    if args.save_png:
        gui.fig.savefig(args.save_png, dpi=150, bbox_inches='tight')
        print(f'Saved to {args.save_png}')
    else:
        gui.show()
