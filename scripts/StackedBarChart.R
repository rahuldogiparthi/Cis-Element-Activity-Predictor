library(ggplot2)
library(dplyr)
kre_breakdown <- data.frame(
  Binding_Status = factor(
    c("AP1 + EGR1 Both", "AP1 Only", "EGR1 Only", "Neither"),
    levels = c("Neither", "EGR1 Only", "AP1 Only", "AP1 + EGR1 Both") 
  ),
  Count = c(1941, 215, 1130, 526) 
)
kre_breakdown <- kre_breakdown %>%
  mutate(Percentage = (Count / sum(Count)) * 100,
         Label = paste0(round(Percentage, 1), "% \n(n=", Count, ")"))
ggplot(kre_breakdown, aes(x = "KREs", y = Percentage, fill = Binding_Status)) +
  geom_bar(stat = "identity", width = 0.4, color = "black", size = 0.5) +
  
  geom_text(aes(label = Label), position = position_stack(vjust = 0.5), size = 4.5, fontface = "bold") +
  
  scale_fill_manual(values = c("Neither" = "#999999",         # Light Gray
                               "EGR1 Only" = "#0072B2",       # Light Blue
                               "AP1 Only" = "#D55E00",        # Orange
                               "AP1 + EGR1 Both" = "lightgreen"))+ # Green
  
  theme_classic(base_size = 14) +
  theme(
    axis.text.x = element_text(face = "bold", size = 16),
    legend.title = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 12)
  ) +
  labs(
    title = "AP1 & EGR1 Binding Overlap at KREs",
    y = "KRE Composition (%)",
    x = NULL
  )
