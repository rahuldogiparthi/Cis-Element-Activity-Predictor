# Load ggplot2
library(ggplot2)

# Create the dataframe. Target_Pct and Bg_Pct columns extracted from HOMER files
motif_data <- data.frame(
  Motif = c("CTCF", "CTCFL", "BCL11A", "REST", "ZEB2", "E2A"),
  P_value = c(1e-131, 1e-102, 1e-16, 1e-12, 1e-11, 1e-9),
  Target_Pct = c(22.27, 22.82, 13.12, 1.22, 10.21, 16.74), 
  Bg_Pct = c(6.69, 8.32, 8.07, 0.27, 6.4, 12.31)           
)

# Calculate Fold Enrichment
motif_data$LogP <- -log10(motif_data$P_value)
motif_data$FoldEnrichment <- motif_data$Target_Pct / motif_data$Bg_Pct
motif_data$Motif <- factor(motif_data$Motif, levels = motif_data$Motif[order(motif_data$LogP)])

# Generate Bubble Plot

ggplot(motif_data, aes(x = FoldEnrichment, y = Motif)) +
  geom_point(aes(size = Target_Pct, color = LogP), alpha = 0.9) +
  scale_color_gradient(low = "#4575B4", high = "#D55E00") +
  scale_size_continuous(range = c(4, 10)) +
  theme_classic(base_size = 14) +
  labs(
    title = "Co-factor Motif Enrichment",
    subtitle = "EGR1-Sensitive vs. Insensitive KREs",
    x = "Fold Enrichment (Sensitive / Insensitive)",
    y = "",
    size = "% of EGR1 Sensitive KREs\nwith Co-factor Motif",
    color = "-log10(p-value)"
  ) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text = element_text(color = "black")
  )
