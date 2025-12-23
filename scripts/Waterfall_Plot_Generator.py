import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'KREs.xlsx'  # Replace with your actual file path
sheet_name = 'Kit Activated'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Sort by Score
sorted_df = df.sort_values(by='Score', ascending=True)

# Assign colors: blue for positive scores, red for negative
colors = ['steelblue' if score >= 0 else 'salmon' for score in sorted_df['Score']]

# Plot using index positions but label x-axis with gene names
fig, ax = plt.subplots(figsize=(16, 6))
bars = ax.bar(x=range(len(sorted_df)), height=sorted_df['Score'].values, color=colors, width=1.0)

# Set gene names as x-tick labels
ax.set_xticks(range(len(sorted_df)))
ax.set_xticklabels(sorted_df['Genes'], rotation=90, fontsize=6)

# Customize appearance
ax.set_xlabel('Genes (Sorted by Score)')
ax.set_ylabel('Score')
ax.set_title('Waterfall Plot of KRE Scores by Gene (Sorted by Score)')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(False)
plt.tight_layout()

# Optional: Save the plot to a PDF
plt.savefig('KRE_waterfall_by_gene.pdf', format='pdf')

# Show the plot
plt.show()
