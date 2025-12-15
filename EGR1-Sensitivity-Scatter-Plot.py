import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load dataframe

df = pd.read_csv("EGR1_SCF_log2FC_grid_withIDs.tsv", sep="\t")

# Thresholds
up_thr   = 0.585
down_thr = -0.585
df["EGR1_group"] = "Other"
sensitive_grids = [
    "Scr_UP_KD_NEUTRAL", "Scr_UP_KD_DOWN",
    "Scr_DOWN_KD_NEUTRAL", "Scr_DOWN_KD_UP"
]
insensitive_grids = [
    "Scr_UP_KD_UP",
    "Scr_DOWN_KD_DOWN"
]
df.loc[df["grid"].isin(sensitive_grids), "EGR1_group"] = "EGR1-sensitive"
df.loc[df["grid"].isin(insensitive_grids), "EGR1_group"] = "EGR1-insensitive"
df["EGR1_group"] = pd.Categorical(
    df["EGR1_group"],
    categories=["EGR1-sensitive", "EGR1-insensitive", "Other"],
    ordered=True
)

palette = {
    "EGR1-sensitive":   "#D55E00",  
    "EGR1-insensitive": "#0072B2",  
    "Other":            "#B0B0B0" 
}

# Create Scatter Plot of peaks
plt.figure(figsize=(12, 9))

sns.scatterplot(
    data=df,
    x="log2FC_KD",
    y="log2FC_Scr",
    hue="EGR1_group",
    palette=palette,
    s=14,
    alpha=0.7,
    linewidth=0
)

for x in [0, up_thr, down_thr]:
    plt.axvline(x, linestyle="--" if x == 0 else ":", color="gray", alpha=0.7)

for y in [0, up_thr, down_thr]:
    plt.axhline(y, linestyle="--" if y == 0 else ":", color="gray", alpha=0.7)

# Plot labels for title and axis
plt.xlabel("log2FC (+SCF / −SCF) - sgEGR1", fontsize=14)
plt.ylabel("log2FC (+SCF / −SCF) - sgControl", fontsize=14)
plt.title("EGR1 Sensitivity of KIT/SCF Chromatin Responses", fontsize=16)

plt.legend(
    title="Category",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
    fontsize=12,
    title_fontsize=13
)

sns.despine(offset=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Save plots
#plt.savefig("EGR1_Sensitivity_Scatter_Plot.pdf")
#plt.savefig("EGR1_Sensitivity_Scatter_Plot.png", dpi=1200)
plt.show()

