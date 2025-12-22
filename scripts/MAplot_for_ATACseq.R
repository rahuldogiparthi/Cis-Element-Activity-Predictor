library(ggplot2)

# Read MAnorm-generated MA values table for all peaks
df <- read.table("/data/manorm/Scr_Plus_SCF_vs_Scr_Minus_SCF_all_MAvalues.xls",
                 header = TRUE, sep = "\t",
                 quote = "", comment.char = "",
                 stringsAsFactors = FALSE, check.names = FALSE)

# Selection metrics for the graph
Acol <- "A_value"
Mcol <- "M_value"

# Drop rows other than M and A
plot_df <- na.omit(data.frame(A = A, M = M))

cutoff <- 0.585
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

sp
