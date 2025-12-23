import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Inputs acquired from merged datasheets between Log-Odds Ratio & KREs, ATAC-seq Footprint scores. Replaced accordingly for each analysis
data = {
    'TF':     ['CTCF',     'CTCFL',     'KLF16',       'REST'],
    'log_odds': [0.137981408, 0.072510256, 0.121389335, 0.111623931],
    'KREs':     [704,        245,         340,          606]
}
df = pd.DataFrame(data)

fp_scores = {
    'ctcf': 0.17154,
    'ctcfl': 0.08679,
    'klf16': 0.17384,
    'rest': 0.03675
}
df['footprint'] = df['TF'].str.lower().map(fp_scores)

# Sort by Log-Odds Ratio
df = df.sort_values('log_odds', ascending=False).reset_index(drop=True)

# Plot
sns.set(style="white")
fig, ax = plt.subplots(figsize=(10, 6))
vmin, vmax = df['footprint'].min(), df['footprint'].max()
if vmin == vmax:
    vmin, vmax = vmin - 1e-6, vmax + 1e-6
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

scatter = ax.scatter(
    x=df['log_odds'],
    y=pd.Categorical(df['TF'], categories=df['TF'], ordered=True),
    s=(df['KREs'] / df['KREs'].max()) * 1200,
    c=df['footprint'],
    cmap=cmap,
    norm=norm,
    alpha=0.85,
    edgecolors='black',
    linewidths=0.7
)

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Footprint score', fontsize=12)

# KREs legend can be changed as per the dataset in use
legend_steps = [200, 400, 600, 700]
for kre in legend_steps:
    ax.scatter([], [], s=(kre / df['KREs'].max()) * 1200,
               color='gray', alpha=0.6, edgecolors='black', linewidths=0.7,
               label=f'KREs: {kre}')
ax.legend(title='KREs (bubble size)', loc='upper right', frameon=True)

# Labels
ax.set_title('TF Enrichment: log-odds vs TFs (color = footprint score)', fontsize=16)
ax.set_xlabel('log(Odds Ratio)', fontsize=14)
ax.set_ylabel('Transcription Factor', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
sns.despine(ax=ax)
ax.invert_yaxis()

x_min, x_max = df['log_odds'].min(), df['log_odds'].max()
ax.set_xlim(x_min - 0.02, x_max + 0.02)
plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.12)

plt.savefig("TF_bubble_plot_selected_with_footprint_right_colorbar.pdf", bbox_inches='tight')
plt.show()
