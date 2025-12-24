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

Scores_CellType = []
for cell_types in score_df['Blood-CellType']:
    # Assigning scores to CFUE or ProE
    if any(cell_type in scoring_dict_CellType and scoring_dict_CellType[cell_type] == 2 for cell_type in cell_types.split(',')):
        Scores_CellType.append(2)
    
    # Assigning scores for HUDEP-2
    elif any('HUDEP2 Only' in cell_type for cell_type in cell_types.split(',')):
        Scores_CellType.append(0)
    
    # Assigning scores for all other cell types
    else:
        Scores_CellType.append(-1)


# Scoring for ABC-Status
Scores_States = []
for state in score_df['ABC-Type']:
    Scores_States.append(scoring_dict_State[state])

score_df['CellType_Score'] = Scores_CellType
score_df['State_Score'] = Scores_States

# Scoring for Evolutionary conservation and H3K27AC status
score_df['Conservation-Scores'] = score_df['PhyloP'].apply(lambda x: 1 if x > 1 else -1)
score_df['h3k27ac_Score'] = score_df['H3K27AC'].apply(lambda x: 1 if x == 1 else -1)

# Scoring for TF occupancy
score_df['Scores'] = score_df['CellType_Score'] + score_df['State_Score'] + score_df['THAP1']*0.01054761 + score_df['EGR1']*0.07722619 + score_df['JUN']*0.5341085 + score_df['JUND']*0.24352597 + score_df['MXI1']*0.031388287 + score_df['RUNX1']*0.006536656 + score_df['ATF3']*0.0399749 + score_df['PKNOX1']*0.03170987 + score_df['ATF2']*0.048396807 + score_df['CTCF']*0.12805343 + score_df['h3k27ac_Score'] + score_df['Conservation-Scores']
score_df.drop(columns=['CellType_Score','State_Score','Conservation-Scores','h3k27ac_Score'],inplace=True)
score_df.sort_values(by=['Scores'], ascending=False,inplace=True)
#score_df.to_csv("KRE_Scores.csv",index=False)
