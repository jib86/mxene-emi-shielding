# ══════════════════════════════════════════════════════════════════════════════
#  heatmap.py — SE heatmap across all microwave bands and 3d dopants
#   Generates Fig. 2 of the manuscript.
#   Stats sidebar with colored label text:
#   • "Spearman ρ = ..."  in muted forest-green
#   • "Partial ρ = ..."   in muted steel-blue
#   • "Exact permutation" in muted purple
#   • p-values and context in dark (#333)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats as sp_stats

# ── 1. GLOBAL STYLE ───────────────────────────────────────────────────────────
BG = '#FAFAFA'
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Inter', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'figure.facecolor':  BG,
    'axes.facecolor':    BG,
    'savefig.facecolor': BG,
})

cmap_soft = matplotlib.colormaps['cividis']

element_colors = {
    'Sc': '#6A0DAD', 'Ti': '#0047AB', 'V':  '#007FFF', 'Cr': '#008B8B',
    'Mn': '#00A36C', 'Fe': '#228B22', 'Co': '#32CD32', 'Ni': '#DAA520',
    'Cu': '#FF4500', 'Zn': '#0066CC'
}

band_display = {
    'Ka-band': 'K$_a$-band',
    'Ku-band': 'K$_u$-band',
}

# ── 3. DFT DATA ───────────────────────────────────────────────────────────────
data = {
    "Band": [
        "Radio", "L-band", "S-band", "C-band", "X-band",
        "Ku-band", "K-band", "Ka-band", "Q-band", "U-band",
        "V-band", "W-band", "F-band", "D-band",
    ],
    "Freq_Text": [
        "150 MHz", "1.5 GHz", "3.0 GHz", "6.0 GHz", "10.0 GHz",
        "15.0 GHz", "22.25 GHz", "33.25 GHz", "41.5 GHz", "50.0 GHz",
        "62.5 GHz", "92.5 GHz", "115 GHz", "140 GHz",
    ],
    "Freq_Hz": [
        0.15e9, 1.5e9, 3.0e9, 6.0e9, 10.0e9,
        15.0e9, 22.25e9, 33.25e9, 41.5e9, 50.0e9,
        62.5e9, 92.5e9, 115e9, 140e9,
    ],
    "Sc": [32.51,32.48,32.48,32.48,32.48,32.48,32.48,32.48,32.48,32.48,32.48,32.48,32.48,32.48],
    "Ti": [32.56,32.54,32.54,32.54,32.54,32.54,32.54,32.54,32.54,32.54,32.54,32.54,32.54,32.54],
    "V":  [18.30,18.28,18.28,18.28,18.28,18.28,18.28,18.28,18.28,18.28,18.28,18.28,18.28,18.28],
    "Cr": [16.89,16.87,16.87,16.87,16.87,16.87,16.87,16.87,16.87,16.87,16.87,16.87,16.87,16.87],
    "Mn": [13.44,13.43,13.43,13.43,13.43,13.43,13.43,13.43,13.43,13.43,13.43,13.43,13.43,13.43],
    "Fe": [17.76,17.74,17.74,17.74,17.74,17.74,17.74,17.74,17.74,17.74,17.74,17.74,17.74,17.74],
    "Co": [16.49,16.47,16.47,16.47,16.47,16.47,16.47,16.47,16.47,16.47,16.47,16.47,16.47,16.47],
    "Ni": [30.83,30.80,30.80,30.80,30.80,30.80,30.80,30.80,30.80,30.80,30.80,30.80,30.80,30.80],
    "Cu": [34.03,34.00,34.00,34.00,34.00,34.01,34.00,34.01,34.00,34.00,34.00,34.00,34.00,34.00],
    "Zn": [44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90,44.90],
}

df = pd.DataFrame(data)
df = df.sort_values(by="Freq_Hz").reset_index(drop=True)

moments = {
    'Sc': 0.00, 'Ti': 3.71, 'V':  8.28, 'Cr': 17.08, 'Mn': 21.98,
    'Fe': 17.94,'Co': 10.23,'Ni': 6.35, 'Cu':  0.00, 'Zn':  0.00
}
metals_low    = ['Zn', 'Cu', 'Sc', 'Ti', 'Ni']
metals_high   = ['V', 'Co', 'Cr', 'Fe', 'Mn']
metals_sorted = metals_low + metals_high
moment_vals   = [moments[m] for m in metals_sorted]
n_low, n_high = len(metals_low), len(metals_high)
GAP           = 0.15

x_low       = np.arange(n_low,  dtype=float)
x_high      = np.arange(n_high, dtype=float) + n_low + GAP
x_positions = np.concatenate([x_low, x_high])
bar_centers = x_positions + 0.5
XLIM        = (0, n_low + n_high + GAP)

df_mw   = df.copy().reset_index(drop=True)
data_mw = df_mw[metals_sorted]

def spearman_rho(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    def avg_rank(v):
        order = np.argsort(v); ranks = np.empty(n); i = 0
        while i < n:
            j = i
            while j < n-1 and v[order[j+1]] == v[order[j]]: j += 1
            avg = (i+j)/2.0+1
            for k in range(i, j+1): ranks[order[k]] = avg
            i = j+1
        return ranks
    return float(np.corrcoef(avg_rank(x), avg_rank(y))[0,1])

mean_se      = {m: float(data_mw[m].mean()) for m in metals_sorted}
mean_se_vals = [mean_se[m] for m in metals_sorted]
rho_mean     = spearman_rho(moment_vals, mean_se_vals)
n_all        = len(metals_sorted)
t_stat       = rho_mean * np.sqrt(n_all-2) / np.sqrt(1-rho_mean**2)
p_val        = 2 * sp_stats.t.sf(abs(t_stat), df=n_all-2)
p_str        = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
print(f"Spearman \u03c1: {rho_mean:.4f},  {p_str}")

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.09, 9.00), dpi=600, facecolor=BG)
gs  = fig.add_gridspec(
    2, 2,
    height_ratios=[3.2, 14.0],
    width_ratios=[5.5, 1.6],
    hspace=0.04, wspace=0.05,
)
ax_lp    = fig.add_subplot(gs[0, 0])
ax_mw    = fig.add_subplot(gs[1, 0])
ax_stats = fig.add_subplot(gs[:, 1])

# ══════════════════════════════════════════════════════════════════════════════
#  LOLLIPOP
# ══════════════════════════════════════════════════════════════════════════════
sep_lp = n_low + 0.5 + GAP / 2
ax_lp.axvspan(0,      sep_lp,         color='#C9E4F5', alpha=0.50, zorder=0)
ax_lp.axvspan(sep_lp, XLIM[1] + 0.5, color='#F5C9C9', alpha=0.50, zorder=0)
ax_lp.yaxis.grid(True, color='#DDDDDD', linewidth=0.4, alpha=0.75, zorder=1)
ax_lp.set_axisbelow(True)

for metal, xc, val in zip(metals_sorted, bar_centers, moment_vals):
    col = element_colors[metal]
    ax_lp.plot([xc, xc], [0, val], color=col, linewidth=1.4,
               solid_capstyle='round', zorder=3)
    ax_lp.scatter(xc, val, color=col, s=48, zorder=4,
                  edgecolors='white', linewidths=0.7)
    label_y = val + 1.30 if val > 0 else 1.30
    ax_lp.text(xc, label_y, f'{val:.2f}' if val > 0 else '0',
               ha='center', va='bottom',
               fontsize=6.5, fontweight='bold', color=col, zorder=5)

ax_lp.text(0.252, 0.93, 'Low-spin',  transform=ax_lp.transAxes,
           ha='center', va='top', fontsize=8,
           style='italic', fontweight='bold', color='#1A4E6E')
ax_lp.text(0.768, 0.93, 'High-spin', transform=ax_lp.transAxes,
           ha='center', va='top', fontsize=8,
           style='italic', fontweight='bold', color='#8B1A1A')

ax_lp.set_xticks(bar_centers)
ax_lp.set_xticklabels(metals_sorted, fontsize=9, fontweight='bold')
ax_lp.xaxis.tick_top()
ax_lp.tick_params(axis='x', length=0, pad=5, labeltop=True, labelbottom=False)
for tick, metal in zip(ax_lp.get_xticklabels(), metals_sorted):
    tick.set_color(element_colors[metal])
ax_lp.set_ylabel('Magnetic\nMoment ($\\mu_B$)', fontsize=8, labelpad=4, fontweight='bold')
ax_lp.set_xlim(*XLIM); ax_lp.set_ylim(0, 25); ax_lp.set_yticks([0,5,10,15,20])
ax_lp.tick_params(axis='y', labelsize=7)
ax_lp.spines[['top','right','bottom']].set_visible(False)
ax_lp.spines['left'].set_color('#AAAAAA'); ax_lp.spines['left'].set_linewidth(0.6)
ax_lp.set_facecolor(BG)

# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
sns.heatmap(data_mw, cmap=cmap_soft, annot=False, cbar=False,
            linewidths=0.8, linecolor='#e8e8e8', ax=ax_mw, vmin=0, vmax=80)
for coll in ax_mw.collections: coll.remove()

norm = matplotlib.colors.Normalize(vmin=0, vmax=80)
sm   = matplotlib.cm.ScalarMappable(cmap=cmap_soft, norm=norm)

for row_idx in range(len(data_mw)):
    for col_idx, (metal, xp) in enumerate(zip(metals_sorted, x_positions)):
        val   = float(data_mw.iloc[row_idx, col_idx])
        rect  = mpatches.FancyBboxPatch(
            (xp, row_idx), 1.0, 1.0, boxstyle="square,pad=0",
            linewidth=0.5, edgecolor='#e8e8e8', facecolor=sm.to_rgba(val), zorder=2)
        ax_mw.add_patch(rect)
        text_color = 'white' if val < 35 else '#111111'
        ax_mw.text(xp+0.5, row_idx+0.5, f'{val:.1f}',
                   ha='center', va='center',
                   fontsize=7, fontweight='bold', color=text_color, zorder=4)

ax_mw.set_xlim(*XLIM); ax_mw.set_ylim(0, len(data_mw)); ax_mw.invert_yaxis()
ax_mw.set_yticks([]); ax_mw.tick_params(axis='y', length=0, pad=8)
for i, row in df_mw.iterrows():
    disp = band_display.get(row['Band'], row['Band'])
    ax_mw.text(-0.012, i+0.5, disp,
               transform=ax_mw.get_yaxis_transform(),
               ha='right', va='bottom', fontsize=8, fontweight='bold', color='#111111')
    ax_mw.text(-0.012, i+0.5, f"({row['Freq_Text']})",
               transform=ax_mw.get_yaxis_transform(),
               ha='right', va='top', fontsize=6.5, color='#555555')

ax_mw.set_xticks([]); ax_mw.set_ylabel(''); ax_mw.set_xlabel('')
ax_mw.spines[['top','right','bottom','left']].set_visible(False)
ax_mw.set_facecolor(BG)

cbar_ax = inset_axes(ax_mw, width="70%", height="3%", loc='lower center',
                     bbox_to_anchor=(0.05,-0.055,0.9,1),
                     bbox_transform=ax_mw.transAxes, borderpad=0)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.outline.set_linewidth(0.5); cbar.outline.set_edgecolor('#444444')
cbar.set_ticks([0,20,40,60,80])
cbar.set_label('Shielding Effectiveness (%)', labelpad=4, weight='bold', fontsize=8)
cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
cbar.ax.set_xlabel(
    '\u2190 Magnetic penalty  (high spin = low SE)          Low spin = high SE \u2192',
    fontsize=6, color='#666666', labelpad=2)

# ══════════════════════════════════════════════════════════════════════════════
#  STATS SIDEBAR — colored label text only, no block backgrounds
# ══════════════════════════════════════════════════════════════════════════════
for spine in ax_stats.spines.values(): spine.set_visible(False)
ax_stats.set_xticks([]); ax_stats.set_yticks([])
ax_stats.set_facecolor(BG); ax_stats.set_xlim(0,1); ax_stats.set_ylim(0,1)

# ── Geometry ─────────────────────────────────────────────────────────────────
LH        = 0.0165
TITLE_H   = 2 * (7.0 * 1.25) / (72 * 7.74)
PAD       = 0.018
SEP_GAP   = 0.020
BLK_H     = 3 * LH
BLK_GAP   = 0.018
TOTAL_H   = PAD + TITLE_H + SEP_GAP + 3*BLK_H + 2*BLK_GAP + PAD

CTR   = 0.814 / 2
BOX_B = CTR - TOTAL_H / 2
BOX_T = CTR + TOTAL_H / 2
BOX_L, BOX_R = 0.06, 0.94
BOX_W = BOX_R - BOX_L

# ── Outer box ────────────────────────────────────────────────────────────────
ax_stats.add_patch(mpatches.FancyBboxPatch(
    (BOX_L, BOX_B), BOX_W, BOX_T - BOX_B,
    boxstyle='round,pad=0.012',
    transform=ax_stats.transAxes,
    linewidth=0.85, edgecolor='#555555',
    facecolor='white', alpha=0.97, zorder=5, clip_on=False,
))

# ── Bold title ───────────────────────────────────────────────────────────────
TITLE_TOP = BOX_T - PAD
ax_stats.text(0.50, TITLE_TOP, 'Statistical\nSummary',
              transform=ax_stats.transAxes,
              ha='center', va='top',
              fontsize=7.0, fontweight='bold', color='#111111',
              linespacing=1.25, zorder=7, clip_on=False)

# ── Thin separator ───────────────────────────────────────────────────────────
SEP_Y = TITLE_TOP - TITLE_H - 0.006
ax_stats.plot([BOX_L+0.07, BOX_R-0.07], [SEP_Y, SEP_Y],
              transform=ax_stats.transAxes,
              color='#999999', linewidth=0.45, zorder=7, clip_on=False)

# ── Three stat blocks ────────────────────────────────────────────────────────
BLOCKS = [
    {
        'label_color': '#2E7D52',
        'label': 'Spearman \u03c1 = \u22120.93',
        'info':  'p < 0.001\n(mean SE, 14 bands)',
    },
    {
        'label_color': '#2B62A0',
        'label': 'Partial \u03c1 = \u22120.94',
        'info':  'p < 0.001\n(ctrl. for \u03c3\u1d49\u1d9c)',
    },
    {
        'label_color': '#6B52A0',
        'label': 'Exact permutation',
        'info':  'p = 0.008\n(mag. vs non-mag.)',
    },
]

XC      = (BOX_L + BOX_R) / 2
cursor  = SEP_Y - SEP_GAP

for blk in BLOCKS:
    label_y = cursor
    info_y  = label_y - LH

    ax_stats.text(XC, label_y, blk['label'],
                  transform=ax_stats.transAxes,
                  ha='center', va='top',
                  fontsize=6.5, color=blk['label_color'],
                  fontweight='semibold',
                  linespacing=1.40, zorder=7, clip_on=False)

    ax_stats.text(XC, info_y, blk['info'],
                  transform=ax_stats.transAxes,
                  ha='center', va='top',
                  fontsize=6.3, color='#363636',
                  linespacing=1.40, zorder=7, clip_on=False)

    cursor -= BLK_H + BLK_GAP

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_path = 'heatmap.png'
plt.savefig(out_path, format='png', dpi=600,
            bbox_inches='tight', facecolor=BG, edgecolor='none')
print(f"heatmap.py: saved → {out_path}")
