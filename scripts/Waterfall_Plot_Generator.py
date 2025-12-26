import pandas as pd
import matplotlib.pyplot as plt

# Input excel sheet can be replaced with KREs or EGR1 Sensitive KREs list to generate the waterfall plots
df = pd.read_excel('/data/KREs.xlsx', sheet_name='Sheet1')

# Sort by score
sorted_df = df.sort_values(by='Score', ascending=True).reset_index(drop=True)

# Color coding based on Evolutionary Conservation (PhyloP scores)
def phylo_class(p):
    if p > 1:
        return 'Conserved'
    if p < -1:
        return 'Accelerated'
    return 'Neutral'

classes = sorted_df['PhyloP'].apply(phylo_class)

color_map = {
    'Conserved':   'green',
    'Accelerated': 'red',
    'Neutral':     'lightgrey',
}
colors = classes.map(color_map)

xlab_col = 'Regions' if 'Regions' in sorted_df.columns else 'Genes'

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(range(len(sorted_df)), sorted_df['Score'].values, color=colors, width=1.0)

ax.set_xticks(range(len(sorted_df)))
ax.set_xticklabels(sorted_df[xlab_col].astype(str), rotation=90, fontsize=6)

ax.set_xlabel(f'KREs (sorted by Score)')
ax.set_ylabel('Score')
ax.set_title('Waterfall Plot of KRE Scores (colored by PhyloP)')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(False)
plt.tight_layout()

# plt.savefig('KRE_waterfall_by_region_phyloP.pdf', format='pdf', bbox_inches='tight')

plt.show()
