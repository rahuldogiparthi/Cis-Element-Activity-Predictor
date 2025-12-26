# Load libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Inputs acquired from merged datasheets between Log-Odds Ratio & KREs, ATAC-seq Footprint scores. Replaced accordingly for each analysis

data = {
    'TF': [
        'XRCC3', 'MCM5', 'SMC3', 'CCAR2', 'SMAD2', 'SFPQ', 'CTCF',
        'GATA1', 'KLF16', 'REST', 'LDB1', 'CTCFL', 'ZNF280A', 'KDM6A',
        'MCM3', 'SRSF7', 'GATA2', 'ARID2', 'STAG1', 'ZZZ3'
    ],
    'log_odds': [
        0.480603543, 0.479316168, 0.293295604, 0.192278605, 0.191216472,
        0.160893794, 0.137981408, 0.122892453, 0.121389335, 0.111623931,
        0.085996805, 0.072510256, 0.062842678, 0.051358912, 0.036933993,
        0.036933993, 0.026187742, 0.025277006, 0.021150711, 0.002298835
    ],
    'KREs': [
        8, 3, 465, 12, 3, 8, 704, 947, 340, 606, 154, 245,
        29, 168, 9, 9, 206, 173, 31, 59
    ]
}

df = pd.DataFrame(data)

# Add footprint scores

fp_scores = {
    'CTCF': 0.17154,
    'KLF16': 0.17384,
    'REST': 0.03675,
    'CTCFL': 0.08679
}
df['footprint'] = df['TF'].map(fp_scores)   # NaN = N/A

# Sort by Log-Odds Ratio

df = df.sort_values('log_odds', ascending=False).reset_index(drop=True)
sns.set(style="white")
fig, ax = plt.subplots(figsize=(12, 9))
valid_fp = df['footprint'].dropna()
vmin, vmax = valid_fp.min(), valid_fp.max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis
colors = []
NA_COLOR = "#C0C0C0"   # color for missing footprint ("N/A")

for fp in df['footprint']:
    if np.isnan(fp):
        colors.append(NA_COLOR)
    else:
        colors.append(cmap(norm(fp)))

# Bubble Plot 

scatter = ax.scatter(
    x=df['log_odds'],
    y=pd.Categorical(df['TF'], categories=df['TF'], ordered=True),
    s=(df['KREs'] / df['KREs'].max()) * 1000,
    color=colors,
    edgecolors='black',
    alpha=0.85
)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required for colorbar with ScalarMappable
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Footprint score", fontsize=12)

# KRE number range can be adjusted as required

for kre in [10, 100, 300, 600, 900]:
    ax.scatter([], [], s=(kre / df['KREs'].max()) * 1000,
               color='gray', edgecolors='black', alpha=0.6,
               label=f'KREs: {kre}')

# Non-significant and downregulated footprints labelled N/A

ax.scatter([], [], color=NA_COLOR, s=200, edgecolors='black', label='Footprint: N/A')
ax.legend(title="Legend", loc='upper right', frameon=True)
ax.set_title('TF Enrichment: log-odds vs TFs\n(color = footprint score; N/A = gray)', fontsize=16)
ax.set_xlabel('log(Odds Ratio)', fontsize=14)
ax.set_ylabel('Transcription Factor', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
sns.despine(ax=ax)
ax.invert_yaxis()
x_min, x_max = df['log_odds'].min(), df['log_odds'].max()
ax.set_xlim(x_min - 0.05, x_max + 0.05)
plt.subplots_adjust(left=0.30, right=0.88, top=0.92, bottom=0.1)
plt.savefig("TF_bubble_plot_with_NA_and_footprint.pdf", bbox_inches='tight')
plt.show()
