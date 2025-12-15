# Load Libraries

import pandas as pd
import numpy as np
import os

# Load the dataset with BPM values for the masterlist of peaks
df = pd.read_csv("EGR1_SCF_BPM.tab", sep="\t", comment="#", header=None)
df.columns = ["chr", "start", "end", "ScrpSCF", "ScrmSCF", "Sg2pSCF", "Sg2mSCF"]

# Add peak IDs
df["peak_id"] = [f"peak{i}" for i in range(1, len(df) + 1)]
df = df[["chr", "start", "end", "peak_id", "ScrpSCF", "ScrmSCF", "Sg2pSCF", "Sg2mSCF"]]

# Compute log2 fold-changes
df["log2FC_Scr"] = np.log2(df["ScrpSCF"] / df["ScrmSCF"])
df["log2FC_KD"]  = np.log2(df["Sg2pSCF"] / df["Sg2mSCF"])

up_thr   = 0.585   # 1.5-fold up
down_thr = -0.585  # 1.5-fold down

def call_state(x):
    if x >= up_thr:
        return "UP"
    elif x <= down_thr:
        return "DOWN"
    else:
        return "NEUTRAL"

df["state_Scr"] = df["log2FC_Scr"].apply(call_state)
df["state_KD"]  = df["log2FC_KD"].apply(call_state)

# Labels for the peaks
df["grid"] = "Scr_" + df["state_Scr"] + "_KD_" + df["state_KD"]

print("\nGrid counts (3×3 states):")
print(df["grid"].value_counts().sort_index())

df["EGR1_sensitivity"] = "other"

# EGR1-sensitive activation: UP in Control, but not activated in EGR1-KD
sens_act = ["Scr_UP_KD_NEUTRAL", "Scr_UP_KD_DOWN"]
df.loc[df["grid"].isin(sens_act), "EGR1_sensitivity"] = "EGR1_sensitive_activation"

# EGR1-sensitive repression: DOWN in Control, but not repressed in EGR1-KD
sens_rep = ["Scr_DOWN_KD_NEUTRAL", "Scr_DOWN_KD_UP"]
df.loc[df["grid"].isin(sens_rep), "EGR1_sensitivity"] = "EGR1_sensitive_repression"

# EGR1-insensitive activation: UP in both
ins_act = ["Scr_UP_KD_UP"]
df.loc[df["grid"].isin(ins_act), "EGR1_sensitivity"] = "EGR1_insensitive_activation"

# EGR1-insensitive repression: DOWN in both
ins_rep = ["Scr_DOWN_KD_DOWN"]
df.loc[df["grid"].isin(ins_rep), "EGR1_sensitivity"] = "EGR1_insensitive_repression"

# EGR1-insensitive baseline: NEUTRAL in both
ins_base = ["Scr_NEUTRAL_KD_NEUTRAL"]
df.loc[df["grid"].isin(ins_base), "EGR1_sensitivity"] = "EGR1_insensitive_baseline"

print("\nEGR1_sensitivity category counts:")
print(df["EGR1_sensitivity"].value_counts())


df.to_csv("EGR1_SCF_log2FC_grid_withIDs.tsv", sep="\t", index=False)
print("\nWrote EGR1_SCF_log2FC_grid_withIDs.tsv")