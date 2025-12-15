# Load Libraries
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(org.Hs.eg.db)

# Input peak files
peak <- readPeakFile("Kit_Activated_Cistrome_TF_selected_hg19_only_coordinates.bed")
txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
peakAnno <- annotatePeak(peak, TxDb=txdb, tssRegion=c(-1000, 1000))

# Plot Pie chart showing genome compostion
plotAnnoPie(peakAnno)
