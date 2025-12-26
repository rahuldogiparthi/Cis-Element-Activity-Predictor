# Load Libraries
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg19.knownGene) # Change to library(TxDb.Hsapiens.UCSC.hg38.knownGene) if using hg38
library(org.Hs.eg.db)

# Input peak files
peak <- readPeakFile("data/PlotAnnoPie/Kit_Activated_Peaks.bed") # Change the peak file accordingly for different conditions
txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
peakAnno <- annotatePeak(peak, TxDb=txdb, tssRegion=c(-1000, 1000))

# Plot Pie chart showing genome compostion
pdf("Kit_Activated_Peaks_AnnoPieChart.pdf")
plotAnnoPie(peakAnno)
dev.off()


