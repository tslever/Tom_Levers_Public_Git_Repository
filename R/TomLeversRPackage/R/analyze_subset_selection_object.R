#' @export
analyze_subset_selection_object <- function(subset_selection_object) {
 summary_for_subset_selection_object <- summary(object = subset_selection_object)
 Mallows_Cp <- summary_for_subset_selection_object$cp
 index_of_model_with_minimum_Mallows_Cp <- which.min(Mallows_Cp)
 coefficients_by_Mallows_Cp <- coef(
  subset_selection_object, index_of_model_with_minimum_Mallows_Cp
 )
 Schwartz_BIC <- summary_for_subset_selection_object$bic
 index_of_model_with_minimum_Schwartz_BIC <- which.min(Schwartz_BIC)
 coefficients_by_Schwartz_BIC <- coef(
  subset_selection_object, index_of_model_with_minimum_Schwartz_BIC
 )
 adjusted_R2 <- summary_for_subset_selection_object$adjr2
 index_of_model_with_maximum_adjusted_R2 <- which.max(adjusted_R2)
 coefficients_by_adjusted_R2 <- coef(
  subset_selection_object, index_of_model_with_maximum_adjusted_R2
 )
 plot(Mallows_Cp, xlab = "Number Of Variables", ylab = "C_p", type = "l")
 index_of_minimum_Mallows_Cp <- which.min(Mallows_Cp)
 minimum_Mallows_Cp <- Mallows_Cp[index_of_minimum_Mallows_Cp]
 points(
  index_of_minimum_Mallows_Cp,
  minimum_Mallows_Cp,
  col = "red",
  cex = 2,
  pch = 20
 )
 plot(Schwartz_BIC, xlab = "Number Of Variables", ylab = "Schwartz BIC", type = "l")
 index_of_minimum_Schwartz_BIC <- which.min(Schwartz_BIC)
 minimum_Schwartz_BIC <- Schwartz_BIC[index_of_minimum_Schwartz_BIC]
 points(
  index_of_minimum_Schwartz_BIC,
  minimum_Schwartz_BIC,
  col = "red",
  cex = 2,
  pch = 20
 )
 plot(adjusted_R2, xlab = "Number Of Variables", ylab = "Adjusted R^2", type = "l")
 index_of_maximum_adjusted_R2 <- which.max(adjusted_R2)
 maximum_adjusted_R2 <- adjusted_R2[index_of_maximum_adjusted_R2]
 points(
  index_of_maximum_adjusted_R2,
  maximum_adjusted_R2,
  col = "red",
  cex = 2,
  pch = 20
 )
 coefficients <- list(
  coefficients_by_Mallows_Cp = coefficients_by_Mallows_Cp,
  coefficients_by_Schwartz_BIC = coefficients_by_Schwartz_BIC,
  coefficients_by_adjusted_R2 = coefficients_by_adjusted_R2
 )
 return(coefficients)
}