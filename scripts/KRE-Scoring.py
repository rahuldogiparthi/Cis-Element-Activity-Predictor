# Load library
import pandas as pd

# Read the KREs file which has been annotated by the identified Kit sensitive features, cell types and chromatin states
score_df = pd.read_table("/data/KRE_Before_Scored.bed")

# Check and remove for any duplicates in the dataset
def remove_duplicates(cell_value):
    unique_values = ','.join(sorted(set(cell_value.split(','))))  # Sort for consistent order
    return unique_values

score_df['Blood-CellType'] = score_df['Blood-CellType'].apply(remove_duplicates)
score_df['ABC-Type'].unique()

# Assigning scores to cell types
scoring_dict_CellType = {'CFUE': 2, 'ProE': 2, 'BasoE': -1, 'PolyE': -1, 'OrthoE': -1,'HUDEP2 Only':0}
scoring_dict_State = {'ABC': 1, 'Non-ABC': -1}

