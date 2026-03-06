"""Neural temporal raster display.

Single panel with N time-bin columns and population bands on Y-axis.
Each column represents a neural event (input arrival, feedforward burst,
or disinhibition).  Each row band is a neuron population.

NeuralRasterGUI accepts optional keyword arguments to override the default
3-word population/bin layout for longer phrases.  Pass the *_4W constants
(or a custom dict) to display 4-word timelines.
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Layout constants (inlined; no dependency on raster-sim/config.py)
ROW_HEIGHT           = 1.8
GROUP_PADDING        = 3.0
ALPHA_MIN            = 0.3
ALPHA_MAX            = 1.0
FONT_SIZE_SPIKE      = 9
FONT_SIZE_ROLE_LABEL = 10
FONT_SIZE_AXIS       = 8


# ── Spike event dataclass (shared by neural_demo.py and timeline_parse.py) ──

@dataclass
class TemporalSpikeEvent:
    """One spike in the neural temporal raster."""
    time_bin: int      # column index (0 = T0, 1 = T1, …)
    word: str
    score: float       # 0–1, controls alpha
    population: str    # e.g. 'INPUT' | 'FF' | 'DIS' | 'COMP_AB' | 'COMP_A' | …
    rank: int          # y position within population (0 = highest score)
    y: float = field(default=0.0, repr=False)   # assigned by NeuralRasterGUI


# ── 3-word population definitions (Y-axis order: top → bottom) ──────────────

POPULATIONS = ['INPUT', 'FF', 'DIS', 'COMP_AB', 'COMP_A', 'FF_AB', 'FF_A']

POP_LABELS = {
    'INPUT':   'Input',
    'FF':      'Feedforward\n(right-context)',
    'DIS':     'Disinhibition\n(left-context\nrebound)',
    'COMP_AB': 'AB|C parse\n(2-word left)',
    'COMP_A':  'A|BC parse\n(1-word left)',
    'FF_AB':   '→FF (AB|C)\nC continuation',
    'FF_A':    '→FF (A|BC)\nBC continuation',
}

POP_COLORS = {
    'INPUT':   '#444444',   # dark gray
    'FF':      '#d06010',   # orange  — excitatory feedforward
    'DIS':     '#1060c0',   # blue    — disinhibition rebound
    'COMP_AB': '#20a040',   # green   — AB|C parse winner
    'COMP_A':  '#9040c0',   # purple  — A|BC parse winner
    'FF_AB':   '#20a040',   # green   — FF continuation of AB|C parse
    'FF_A':    '#9040c0',   # purple  — FF continuation of A|BC parse
}

POP_TOP_N = {
    'INPUT':   3,
    'FF':      8,
    'DIS':     8,
    'COMP_AB': 8,
    'COMP_A':  8,
    'FF_AB':   8,
    'FF_A':    8,
}

# ── 3-word time-bin column labels ────────────────────────────────────────────
#
# Eight-bin timeline: two FF→DIS cycles for first pair, then C arrives and the
# DIS step using C's left-context IS the competition, followed by FF continuations.
# FF→DIS cycle ≈ 100ms; word arrivals ≈ 400ms → ~4 cycles converge before next word.
#
#   T0: first word arrives (t=0)
#   T1: →FF cycle 1 — right-contexts of T0 word scored against second input (+400ms)
#   T2: ←DIS cycle 1 — left-contexts of T1 burst → refined T0-word class (+100ms)
#   T3: →FF cycle 2 — right-contexts of T2 class → refined T1-word subs (+100ms)
#   T4: ←DIS cycle 2 — converged T0-word class (~400ms elapsed) (+100ms)
#   T5: →FF — third input arrives; right-contexts of T4 → T1-word subs (+100ms)
#   T6: ←DIS *C — C's left-context disinhibits left antecedent (+100ms)
#               = competition: *C → A'+B' (AB|C) or *(B'+C) → A' (A|BC)
#               The three-element asymmetry makes this step competitive.
#   T7: →FF * — right-contexts of competition winners → next-word predictions (+100ms)
#               FF_AB: C continuation (1-word); FF_A: BC continuation (2-word)

N_TIME_BINS = 8

TIME_BIN_LABELS = [
    't=0\n"we"',
    '+400ms\n→FF "should"\nA* + B',
    '+100ms\n←DIS "we"',
    '+100ms\n→FF "should"\nrefined',
    '+100ms\n←DIS "we"\nconverged',
    '+100ms\n→FF "go"\n(A*+B)*+C',
    '+100ms\n←DIS *C\nA\'+B\' | A\'',
    '+100ms\n→FF *\nAB|C / A|BC',
]

TIME_BIN_COLORS = [
    '#888888',              # T0 input
    POP_COLORS['FF'],       # T1 →FF should
    POP_COLORS['DIS'],      # T2 ←DIS we
    POP_COLORS['FF'],       # T3 →FF should (cycle 2)
    POP_COLORS['DIS'],      # T4 ←DIS we (converged)
    POP_COLORS['FF'],       # T5 →FF go
    POP_COLORS['DIS'],      # T6 ←DIS *C (competition)
    POP_COLORS['FF'],       # T7 →FF continuation
]


# ── 4-word extension: additional populations for T9–T10 ──────────────────────
#
# When a fourth word D arrives, D's left-context drives a second competition:
#   COMP_L3  (ABC|D): 3-word substitutes for ABC — 3-step cascade from left_3
#   COMP_L2  (AB|CD): 2-word substitutes for AB  — reactivation of T6 COMP_AB
#   COMP_L1  (A|BCD): 1-word substitutes for A   — reactivation of T6 COMP_A
#   FF_R1    (ABC|D): 1-word D continuation
#   FF_R2    (AB|CD): 2-word CD continuation
#   FF_R3    (A|BCD): 3-word BCD continuation

POPULATIONS_4W = POPULATIONS + ['COMP_L3', 'COMP_L2', 'COMP_L1', 'FF_R1', 'FF_R2', 'FF_R3']

POP_LABELS_4W = {
    **POP_LABELS,
    'COMP_L3': 'ABC|D parse\n(3-word left)',
    'COMP_L2': 'AB|CD parse\n(2-word left)',
    'COMP_L1': 'A|BCD parse\n(1-word left)',
    'FF_R1':   '→FF (ABC|D)\nD continuation',
    'FF_R2':   '→FF (AB|CD)\nCD continuation',
    'FF_R3':   '→FF (A|BCD)\nBCD continuation',
}

POP_COLORS_4W = {
    **POP_COLORS,
    'COMP_L3': '#108060',   # dark teal  — 3-word left (new)
    'COMP_L2': '#30b068',   # medium green (lighter than COMP_AB)
    'COMP_L1': '#7030b0',   # medium purple (lighter than COMP_A)
    'FF_R1':   '#108060',   # dark teal (same as COMP_L3)
    'FF_R2':   '#30b068',   # medium green (same as COMP_L2)
    'FF_R3':   '#7030b0',   # medium purple (same as COMP_L1)
}

POP_TOP_N_4W = {
    **POP_TOP_N,
    'COMP_L3': 8,
    'COMP_L2': 8,
    'COMP_L1': 8,
    'FF_R1':   8,
    'FF_R2':   8,
    'FF_R3':   8,
}

N_TIME_BINS_4W = 11

TIME_BIN_LABELS_4W = TIME_BIN_LABELS + [
    '+100ms\n→FF "hat"\n(ABC)*+D',
    '+100ms\n←DIS *D\nABC|D/AB|CD/A|BCD',
    '+100ms\n→FF **\nABC|D/AB|CD/A|BCD',
]

TIME_BIN_COLORS_4W = TIME_BIN_COLORS + [
    POP_COLORS['FF'],    # T8  →FF D arrives
    POP_COLORS['DIS'],   # T9  ←DIS *D (4-word competition)
    POP_COLORS['FF'],    # T10 →FF ** (4-word continuations)
]


def _score_to_alpha(score):
    s = max(0.0, min(1.0, score))
    return ALPHA_MIN + s * (ALPHA_MAX - ALPHA_MIN)


class NeuralRasterGUI:
    """Temporal raster: N time bins × population bands.

    Each column is one neural time step.  Each row group is one
    neuron population.  Word labels are the neuron identities.

    Pass the *_4W module constants (or custom dicts) via keyword arguments
    to display 4-word or longer timelines:

        gui = NeuralRasterGUI(events, title=...,
                              populations=POPULATIONS_4W,
                              pop_labels=POP_LABELS_4W,
                              pop_colors=POP_COLORS_4W,
                              pop_top_n=POP_TOP_N_4W,
                              n_time_bins=N_TIME_BINS_4W,
                              time_bin_labels=TIME_BIN_LABELS_4W,
                              time_bin_colors=TIME_BIN_COLORS_4W)
    """

    def __init__(self, events, title: str = "", *,
                 populations=None, pop_labels=None, pop_colors=None, pop_top_n=None,
                 n_time_bins=None, time_bin_labels=None, time_bin_colors=None):
        self.events = events
        self.title  = title

        # Population config — fall back to 3-word module defaults
        self.populations     = populations     if populations     is not None else POPULATIONS
        self.pop_labels      = pop_labels      if pop_labels      is not None else POP_LABELS
        self.pop_colors      = pop_colors      if pop_colors      is not None else POP_COLORS
        self.pop_top_n       = pop_top_n       if pop_top_n       is not None else POP_TOP_N

        # Time-bin config
        self.n_time_bins     = n_time_bins     if n_time_bins     is not None else N_TIME_BINS
        self.time_bin_labels = time_bin_labels if time_bin_labels is not None else TIME_BIN_LABELS
        self.time_bin_colors = time_bin_colors if time_bin_colors is not None else TIME_BIN_COLORS

        self._pop_y_base   = {}
        self._pop_y_ranges = {}
        self._assign_y_layout()

        for ev in self.events:
            base = self._pop_y_base.get(ev.population, 0.0)
            ev.y = base + ev.rank * ROW_HEIGHT

        self._build_figure()
        self._draw()

    # ── Y layout ────────────────────────────────────────────────────

    def _assign_y_layout(self):
        y = 0.0
        for pop in self.populations:
            self._pop_y_base[pop] = y
            n = self.pop_top_n.get(pop, 8)
            group_height = (n - 1) * ROW_HEIGHT
            self._pop_y_ranges[pop] = (y, y + group_height)
            y += group_height + GROUP_PADDING
        self._max_y = y - GROUP_PADDING

    # ── Figure ──────────────────────────────────────────────────────

    def _build_figure(self):
        fig_w = max(16, self.n_time_bins * 2.5)
        self.fig = plt.figure(figsize=(fig_w, 11))
        if self.title:
            self.fig.suptitle(self.title, fontsize=12, fontweight='bold', y=0.98)

        gs = GridSpec(2, 1, figure=self.fig,
                      height_ratios=[8, 1],
                      hspace=0.15, top=0.93, bottom=0.04,
                      left=0.16, right=0.97)

        self.ax = self.fig.add_subplot(gs[0])

        ax_legend = self.fig.add_subplot(gs[1])
        ax_legend.set_axis_off()
        ax_legend.text(
            0.5, 0.7,
            'Feedforward (orange): fast — right-context synapses fire on input coincidence'
            '   •   '
            'Disinhibition (blue): ~200ms slower — left-context rebound refines both sides',
            ha='center', va='center', fontsize=8, color='#555555',
            transform=ax_legend.transAxes,
        )
        ax_legend.text(
            0.5, 0.2,
            'Two FF→DIS cycles complete before competition  —  '
            'parse winners (green/purple) feed FF continuation',
            ha='center', va='center', fontsize=8, color='#555555',
            transform=ax_legend.transAxes,
        )

    # ── Drawing ─────────────────────────────────────────────────────

    def _draw(self):
        ax = self.ax
        ax.set_xlim(-0.1, self.n_time_bins + 0.1)
        ax.set_ylim(self._max_y + ROW_HEIGHT * 0.5, -ROW_HEIGHT * 6.0)
        ax.set_xticks([])
        ax.set_yticks([])

        # ── Column backgrounds + time-bin labels ──
        for tb in range(self.n_time_bins):
            col = self.time_bin_colors[tb]
            ax.axvspan(tb, tb + 1, alpha=0.04, color=col, zorder=0)
            ax.axvline(x=tb, color='#cccccc', linewidth=0.8, zorder=0)
            ax.text(tb + 0.5, -ROW_HEIGHT * 4.8,
                    self.time_bin_labels[tb],
                    fontsize=6.5, color=col,
                    ha='center', va='bottom', linespacing=1.3,
                    fontweight='bold')
        ax.axvline(x=self.n_time_bins, color='#cccccc', linewidth=0.8, zorder=0)

        # ── Population bands + Y-axis labels ──
        for pop in self.populations:
            y_lo, y_hi = self._pop_y_ranges[pop]
            color = self.pop_colors[pop]
            ax.axhspan(y_lo - ROW_HEIGHT * 0.35, y_hi + ROW_HEIGHT * 0.35,
                       alpha=0.06, color=color, zorder=0)
            if pop != self.populations[-1]:
                ax.axhline(y=y_hi + ROW_HEIGHT * 0.35 + GROUP_PADDING / 2,
                           color='#eeeeee', linewidth=0.5, zorder=0)

            label_y = (y_lo + y_hi) / 2
            ax.text(-0.05, label_y, self.pop_labels[pop],
                    fontsize=FONT_SIZE_ROLE_LABEL - 1, color=self.pop_colors[pop],
                    fontweight='bold', va='center', ha='right', clip_on=False)

        # ── Spike words ──
        for ev in self.events:
            color = self.pop_colors.get(ev.population, '#333333')
            alpha = _score_to_alpha(ev.score)
            col_x = ev.time_bin + 0.05
            ax.text(col_x, ev.y, ev.word,
                    fontsize=FONT_SIZE_SPIKE, color=color, alpha=alpha,
                    ha='left', va='center', clip_on=True,
                    fontfamily='monospace')

        # ── Timing arrows between columns ──
        arrow_kw = dict(arrowstyle='->', color='#aaaaaa', lw=1.0,
                        mutation_scale=10)
        timing = ['400ms'] + ['~100ms'] * (self.n_time_bins - 1)
        for tb in range(self.n_time_bins - 1):
            mid_y = -ROW_HEIGHT * 1.8
            ax.annotate('', xy=(tb + 1 + 0.05, mid_y),
                        xytext=(tb + 0.95, mid_y),
                        arrowprops=arrow_kw)
            ax.text(tb + 0.5, mid_y - ROW_HEIGHT * 0.6,
                    timing[tb], fontsize=6, color='#aaaaaa', ha='center')

    def show(self):
        plt.show()
