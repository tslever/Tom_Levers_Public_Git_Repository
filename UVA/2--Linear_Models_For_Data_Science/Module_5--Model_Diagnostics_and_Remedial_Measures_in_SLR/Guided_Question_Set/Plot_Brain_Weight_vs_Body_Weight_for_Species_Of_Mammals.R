library(MASS)
library(dplyr)
library(Dict)
library(ggplot2)
library(TomLeversRPackage)

species_body_weight_and_brain_weight <- 
    MASS::mammals %>% rename(body_weight = body, brain_weight = brain)
head(species_body_weight_and_brain_weight, n = 3)

brain_weight_thresholds_and_data_subsets <- dict(
    'Maximum Weight' = species_body_weight_and_brain_weight,
    '1000 g' = species_body_weight_and_brain_weight %>% filter(brain_weight < 1000),
    '100 g' = species_body_weight_and_brain_weight %>% filter(brain_weight < 100),
    '25 g' = species_body_weight_and_brain_weight %>% filter(brain_weight < 25),
    '5 g' = species_body_weight_and_brain_weight %>% filter(brain_weight < 5)
)

dev.new()
plot(
    ggplot(
        brain_weight_thresholds_and_data_subsets$get('1000 g'),
        aes(x = body_weight, y = brain_weight)
    ) +
        geom_point(alpha = 0.2) +
        geom_smooth(method = "lm", se = FALSE) +
        labs(
            x = "body weight (kg)",
            y = "brain weight (g)",
            title = paste(
                "Brain Weight vs. Body Weight for Species of Land Mammals\n",
                "with Brain Weights Less Than ",
                "1000 g",
                sep = ""
            )
       ) +
       theme(
           plot.title = element_text(hjust = 0.5),
           axis.text.x = element_text(angle = 0)
       )
)