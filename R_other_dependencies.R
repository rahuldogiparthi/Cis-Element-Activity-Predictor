#!/usr/bin/env Rscript

# R_other_dependencies.R
# This script installs the specific GitHub-hosted dependencies used in the analysis.

# Load devtools
library(devtools)

# Install BuenColors
# Source: https://github.com/caleblareau/BuenColors
devtools::install_github("caleblareau/BuenColors", upgrade = "never")

# Install gchromVAR
# Source: https://github.com/caleblareau/gchromVAR
devtools::install_github("caleblareau/gchromVAR", upgrade = "never")
