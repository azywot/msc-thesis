#!/usr/bin/env python3
"""
Orchestrator / sub-agent scaling figure (two-row grouped bar chart).

Row 1: Orchestrator scaling – bars averaged over sub-agent sizes, whiskers = sub-agent min/max.
Row 2: Sub-agent scaling   – bars averaged over orchestrator sizes, whiskers = orchestrator min/max.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data/results/plots/orchestrator_capabilities"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# data[benchmark][orchestrator_size][thinking_mode] = [1.7B, 8B, 32B sub-agent]
data = {
    'GAIA': {
        '8B':  {'No': [7.9, 12.7, 13.3], 'Orch': [23.0, 23.0, 23.6], 'All': [18.2, 24.8, 24.8]},
        '32B': {'No': [15.2, 18.8, 15.8], 'Orch': [20.6, 21.8, 23.0], 'All': [22.4, 21.2, 23.6]},
    },
    'GPQA': {
        '8B':  {'No': [41.9, 42.4, 46.0], 'Orch': [59.1, 58.6, 58.6], 'All': [59.1, 58.6, 59.1]},
        '32B': {'No': [50.5, 48.5, 55.1], 'Orch': [63.1, 63.6, 64.6], 'All': [64.1, 64.1, 63.1]},
    },
    'AIME': {
        '8B':  {'No': [23.3, 20.0, 31.7], 'Orch': [56.7, 55.0, 56.7], 'All': [56.7, 55.0, 56.7]},
        '32B': {'No': [21.7, 23.3, 35.0], 'Orch': [51.7, 53.3, 55.0], 'All': [53.3, 58.3, 58.3]},
    },
    'MuSiQue': {
        '8B':  {'No': [9.5, 11.0, 11.0], 'Orch': [17.5, 14.0, 16.5], 'All': [15.5, 15.0, 17.5]},
        '32B': {'No': [13.5, 13.5, 16.0], 'Orch': [18.5, 19.0, 17.0], 'All': [20.5, 17.0, 20.0]},
    },
    'HLE': {
        '8B':  {'No': [3.0, 4.5, 2.5], 'Orch': [2.0, 4.0, 2.0], 'All': [2.5, 4.0, 2.0]},
        '32B': {'No': [3.0, 6.0, 4.5], 'Orch': [4.0, 4.0, 4.0], 'All': [3.5, 3.5, 3.5]},
    },
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10.5,
    'axes.labelsize': 9,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

benchmarks = ['GAIA', 'GPQA', 'AIME', 'MuSiQue', 'HLE']
thinking_modes = ['No', 'Orch', 'All']
thinking_labels = ['No\nThink', 'Orch\nThink', 'All\nThink']
sub_sizes_idx = {'1.7B': 0, '8B': 1, '32B': 2}

orch_colors = {'8B': '#6BAED6', '32B': '#FD8D3C'}
sub_colors = {'1.7B': '#C7E9C0', '8B': '#74C476', '32B': '#238B45'}

fig, axes = plt.subplots(2, 5, figsize=(13.2, 5.4), sharex='col')

x = np.arange(len(thinking_modes))

# ── Row 1: orchestrator scaling ───────────────────────────────────────────────
bar_width_r1 = 0.36
for i, bench in enumerate(benchmarks):
    ax = axes[0, i]
    for j, orch in enumerate(['8B', '32B']):
        means, mins, maxs = [], [], []
        for tm in thinking_modes:
            vals = data[bench][orch][tm]
            means.append(np.mean(vals))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
        means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
        yerr = np.vstack([means - mins, maxs - means])
        offset = (j - 0.5) * bar_width_r1
        ax.bar(x + offset, means, bar_width_r1, yerr=yerr,
               color=orch_colors[orch], edgecolor='black', linewidth=0.4,
               error_kw=dict(lw=0.7, capsize=2.5, capthick=0.7, ecolor='#333'),
               label=f'{orch} orchestrator')
    ax.set_title(bench, fontweight='bold', pad=5)
    ax.grid(axis='y', alpha=0.35, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    ax.tick_params(axis='x', length=0)
    if i == 0:
        ax.set_ylabel('Avg. accuracy [%]')

# ── Row 2: sub-agent scaling ──────────────────────────────────────────────────
bar_width_r2 = 0.26
for i, bench in enumerate(benchmarks):
    ax = axes[1, i]
    for k, sub in enumerate(['1.7B', '8B', '32B']):
        idx = sub_sizes_idx[sub]
        means, mins, maxs = [], [], []
        for tm in thinking_modes:
            vals = [data[bench]['8B'][tm][idx], data[bench]['32B'][tm][idx]]
            means.append(np.mean(vals))
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
        means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
        yerr = np.vstack([means - mins, maxs - means])
        offset = (k - 1) * bar_width_r2
        ax.bar(x + offset, means, bar_width_r2, yerr=yerr,
               color=sub_colors[sub], edgecolor='black', linewidth=0.4,
               error_kw=dict(lw=0.7, capsize=2.2, capthick=0.7, ecolor='#333'),
               label=f'{sub} sub-agent')
    ax.grid(axis='y', alpha=0.35, linewidth=0.5, linestyle=':')
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(thinking_labels)
    ax.tick_params(axis='x', length=0)
    if i == 0:
        ax.set_ylabel('Avg. accuracy [%]')

fig.text(-0.005, 0.74, 'Scaling the\norchestrator',
         rotation=90, va='center', ha='center',
         fontsize=10, fontweight='bold')
fig.text(-0.005, 0.28, 'Scaling the\nsub-agent',
         rotation=90, va='center', ha='center',
         fontsize=10, fontweight='bold')

handles1, labels1 = axes[0, 0].get_legend_handles_labels()
fig.legend(handles1, labels1, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.025), frameon=False,
           columnspacing=2.0, handlelength=1.3)

handles2, labels2 = axes[1, 0].get_legend_handles_labels()
fig.legend(handles2, labels2, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 0.515), frameon=False,
           columnspacing=2.0, handlelength=1.3)

fig.text(0.5, -0.02,
         'Whiskers show min–max across the held-out dimension '
         '(sub-agent size in row 1, orchestrator size in row 2).',
         ha='center', va='top', fontsize=8, style='italic', color='#444')

plt.tight_layout(rect=[0.015, 0.005, 1, 0.97])
plt.subplots_adjust(hspace=0.55)

out_pdf = OUT_DIR / "orchestrator_subagent_scaling.pdf"
out_png = OUT_DIR / "orchestrator_subagent_scaling.png"
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, bbox_inches='tight', dpi=220)
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")
