library(ggplot2)

# Read MAnorm-generated MA values table.
df <- read.table("/data/manorm/sgControl_SCF_vs_sgControl_PBS_MAvalues.xls",
                 header = TRUE, sep = "\t",
                 quote = "", comment.char = "",
                 stringsAsFactors = FALSE, check.names = FALSE)

# Selection metrics for the graph
Acol <- "A_value"
Mcol <- "M_value"

# Drop rows other than M and A
plot_df <- na.omit(data.frame(A = A, M = M))

cutoff <- 0.585 # Fold Change cut off at 1.5 in Log2 scale
# Divide the peaks into classes based on fold change cutoffs
plot_df$Class <- "Neutral"
plot_df$Class[plot_df$M >  cutoff] <- "Kit Activated"
plot_df$Class[plot_df$M < -cutoff] <- "Kit Repressed"
plot_df$Class <- factor(plot_df$Class, levels = c("Kit Activated", "Kit Repressed", "Neutral"))

# Plot
sp <- ggplot(plot_df, aes(x = A, y = M, color = Class)) +
  geom_point(size = 0.6, alpha = 0.7) +
  geom_hline(yintercept = c(-cutoff, cutoff), linewidth = 0.5) +
  scale_color_manual(values = c("Kit Activated" = "red",
                                "Kit Repressed" = "blue",
                                "Neutral"       = "grey70")) +
  labs(x = "Average Signal Intensity (A)",
       y = "Fold Change (M)") +
  theme_classic(base_size = 12) +
  theme(legend.title = element_blank())
print(sp)
ggsave("MAnorm_plot.pdf", plot=sp, width = 6, height = 5, units="in")
