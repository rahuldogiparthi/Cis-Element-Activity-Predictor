# Code template obtained from https://rdrr.io/github/caleblareau/gchromVAR/f/vignettes/gchromVAR_vignette.Rmd for gChromVAR analysis
# The input datasets and analysis have been performed with the hg19 assembly to associate with the gchromVAR analysis.
# Load the libraries

library(GenomicRanges)
library(Rsubread)
library(gtools)
library(magrittr)
library(dplyr)
library(ChIPQC)
library(tidyr)
library(ggplot2)
library(chromVAR)
library(gchromVAR)
library(SummarizedExperiment)
library(data.table)
library(BiocParallel)
library(BSgenome.Hsapiens.UCSC.hg19)
library(GenomicAlignments)
library(motifmatchr)
library(JASPAR2024)
library(pheatmap)

# Load .narrowPeak files. Peaks files are available at GSE314033
peaks <- dir("data/narrowpeaks/", pattern = "*.narrowPeak", full.names = TRUE)
peaks<-mixedsort(peaks)
myPeaks <- lapply(peaks, ChIPQC:::GetGRanges, simple = TRUE)
names(myPeaks)<-c("ScrmSCF_1","ScrmSCF_2","ScrpSCF_1","ScrpSCF_2")

#Group<-factor(c(rep("ScrmSCF",1), rep("ScrpSCF",1)))

# Generate consensus counts for the peaks
consensus_counts_generator<-GRangesList(myPeaks)
non_overlapping_region_counts<-function(x){
  reduced <- reduce(unlist(myGRangesList))
  consensusIDs <- paste0("consensus_", seq(1, length(reduced)))
  mcols(reduced) <- do.call(cbind, lapply(myGRangesList, function(x) (reduced %over% x) + 0))
  reducedConsensus <- reduced
  mcols(reducedConsensus) <- cbind(as.data.frame(mcols(reducedConsensus)), consensusIDs)
  consensusIDs <- paste0("consensus_", seq(1, length(reducedConsensus)))
  return(reducedConsensus)
}
myGRangesList<-consensus_counts_generator
consensusToCount<-non_overlapping_region_counts(consensus_counts_generator)
occurrences <- elementMetadata(consensusToCount) %>% as.data.frame %>% dplyr::select(-consensusIDs) %>% rowSums
table(occurrences) %>% rev %>% cumsum
consensusToCount <- consensusToCount[occurrences >= 2, ]

# Feature Counts from bams. Raw fastq files are available at GSE314033

bamsToCount <- dir("data/bams/", full.names = TRUE, pattern = "*.\\.bam$")
bamsToCount<-mixedsort(bamsToCount)
bamsToCount
regionsToCount <- data.frame(GeneID = paste("ID", seqnames(consensusToCount),
                                            start(consensusToCount), end(consensusToCount), sep = "_"), Chr = seqnames(consensusToCount),
                             Start = start(consensusToCount), End = end(consensusToCount), Strand = strand(consensusToCount))
fcResults <- featureCounts(bamsToCount, annot.ext = regionsToCount, isPairedEnd = TRUE,
                           countMultiMappingReads = FALSE, maxFragLength = 100)
myCounts <- fcResults$counts

# gChromVAR analysis                                      
# Mapping counts to Hematopoietic SNP regions

non_overlapping_regions_df<-non_overlapping_region_counts(consensus_counts_generator)
summary_generator <- non_overlapping_regions_df[occurrences >= 2, ]
myCounts <- summarizeOverlaps(summary_generator, bamsToCount, singleEnd = FALSE)
myCounts <- myCounts[rowSums(assay(myCounts)) > 5, ]
myCounts2 <- addGCBias(myCounts, genome = BSgenome.Hsapiens.UCSC.hg19)
files <- list.files(system.file('extdata/paper/PP001/',package='gchromVAR'), full.names = TRUE, pattern = "*.bed$")
ukbb <- importBedScore(rowRanges(myCounts2), files, colidx = 5)
ukbb_wDEV <- computeWeightedDeviations(myCounts2, ukbb)
zscoredf <- reshape2::melt(t(assays(ukbb_wDEV)[["z"]]))
zscoredf[,2] <- gsub("_PP001", "", zscoredf[,2])
colnames(zscoredf) <- c("Sample", "Trait", "Z-score")
write.csv(zscoredf,file="gchromVAR_analysis_table_zscores.csv", row.names=FALSE)

# Visualizing the accessibility variation between conditions at SNPs by Heatmap                                          
df_avg <- zscoredf %>%
  mutate(
    base_sample = sub("_[0-9]+_sorted\\.bam$", "", Sample),
    z = as.numeric(`Z-score`)
  ) %>%
  group_by(base_sample, Trait) %>%
  summarise(z = mean(z, na.rm = TRUE), .groups = "drop")

mat <- df_avg %>%
  tidyr::pivot_wider(names_from = base_sample, values_from = z) %>%
  as.data.frame()

rownames(mat) <- mat$Trait
mat$Trait <- NULL
mat <- as.matrix(mat)

pdf("gChromVAR_heatmap_zscores.pdf", width = 6, height = 8)
pheatmap(mat)
dev.off()
