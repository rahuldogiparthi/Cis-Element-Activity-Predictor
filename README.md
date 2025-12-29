# Predicting *Cis*-Element Responses to Cell Signaling in Red Blood Cell Precursors

## About
Chromatin signatures at *cis*-regulatory elements can predict how they regulate transcription. However, accurate cell type-specific prediction of functional *cis*-elements remains a significant computational challenge due to the lack of context regarding the role of environmental stimuli that may alter activity. Most existing datasets use static molecular and sequence features which do not evaluate *cis*-element responses to extracellular cues. Here, we have developed a multi-factorial strategy (RNA-seq, ATAC-seq, ChIP-seq, Promoter Capture HiC) that identifies, annotates, ranks and functionally tests candidates. We evaluated *cis*-element activity in response to the Kit receptor tyrosine kinase (activated by Stem Cell Factor (SCF)), given its crucial role in hematopoietic and erythroid progenitor cell (EPC) survival and lineage commitment. To investigate chromatin changes mediated by Kit/SCF, we performed ATAC-seq and RNA-seq in acutely SCF stimulated HUDEP-2 cells and mapped enhancer-promoter interactions using existing datasets.

RNA-seq data in acutely Kit-stimulated cells was filtered by Kit-Activated transcripts and annotated for potential nearby or interacting *cis*-elements. We then trained a XGBoost model with Kit sensitive and Kit-insensitive control regions. The overall accuracy of this model at predicting Kit-response loci was 86%. This repository contains the analysis scripts for developing the *cis*-element activity predictor and visualizing data as provided in the manuscript - **Predicting *Cis*-Element Responses to Cell Signaling in Red Blood Cell Precursors**, submitted to *Nucleic Acids Research* (NAR).

> **Note:** This repository provides the analysis workflow for reproducibility and transparency. It is not intended as a standalone software package.

---

## Repository Structure

```text
cis-element-activity-predictor/
│
├── data/                                                 # Contains datasets.txt (Links to GEO and Zenodo)
│
├── scripts/                                              # Main analysis scripts
│   ├── Genome_Annotation.R                               # Genomic annotation of peaks
│   ├── MAplot_for_ATACseq.R                              # Visualization of ATAC-seq peaks
│   ├── gchromVAR_Analysis.R                              # gchromVAR analysis
│   ├── KRE-Scoring.py                                    # KRE scoring
│   ├── Categorizing_EGR1_Sensitivity_Across_Peaks.py     # Annotating EGR1-Sensitive and -Insensitive accessible peaks
|   ├── EGR1-Sensitivity-Scatter-Plot.py                  # Visualization script
│   ├── Bubble_Plot_Generator.py                          # Visualization script
│   ├── Waterfall_Plot_Generator.py                       # Visualization script
│   └── XGBoost-Model/                                    # XGBoost Model scripts
│       ├── Training_and_Evaluation.py                    # Main scripts for XGBoost training and evaluation
│       └── Scatterplot_FeatureImportances.py             # Model interpretation plotting
│
├── ciselementactivitypredictor.yml                       # Conda environment configuration
├── R_other_dependencies.R                                # Helper script for other hosted R packages
└── README.md                                             # Project documentation
