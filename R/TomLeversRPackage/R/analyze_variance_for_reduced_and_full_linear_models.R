#' @title analyze_variance_for_reduced_and_full_linear_models
#' @description Analyzes the variance of reduced and full linear models
#' @param reduced_linear_model A reduced linear model with the same response as a full linear model and a subset of the predictors of the full model
#' @param full_linear_model A full linear model with the same response as a reduced linear model and a superset of the predictors of the reduced model
#' @return The analysis of variance for the linear models
#' @examples analysis_of_variance <- analyze_variance_for_reduced_and_full_linear_models(reduced_linear_model, full_linear_model)

#' @export
analyze_variance_for_reduced_and_full_linear_models <- function(reduced_linear_model, full_linear_model) {
 analysis <- capture.output(anova(reduced_linear_model, full_linear_model))

 test_statistic_F0_for_Partial_F_Test <- (anova(reduced_model, full_model)$"F")[2]
 line_with_test_statistic_F0_for_Partial_F_Test <- paste("\ntest statistic F0 for Partial F Test: ", test_statistic_F0_for_Partial_F_Test, sep = "")
 analysis <- append(analysis, line_with_test_statistic_F0_for_Partial_F_Test)

 significance_level <- 0.05
 number_of_predictors_dropped <- (anova(reduced_model, full_model)$"Df")[2]
 residual_degrees_of_freedom_of_full_model <- (anova(reduced_model, full_model)$"Res.Df")[2]
 critical_value_F <- calculate_critical_value_Fc(significance_level, number_of_predictors_dropped, residual_degrees_of_freedom_of_full_model)
 line_with_critical_value_F <- paste("Fc(alpha = ", significance_level, ", predictors_dropped = ", number_of_predictors_dropped, ", DFRes(full) = ", residual_degrees_of_freedom_of_full_model, ") = ", critical_value_F, sep = "")
 analysis <- append(analysis, line_with_critical_value_F)

 p_value_for_Partial_F_Test <- (anova(reduced_model, full_model)$"Pr(>F)")[2]
 line_with_p_value_for_Partial_F_Test <- paste("P(F > F0) for Partial F Test: ", p_value_for_Partial_F_Test, sep = "")
 analysis <- append(analysis, line_with_p_value_for_Partial_F_Test)

 line_with_significance_level <- paste("significance level: ", significance_level, sep = "")
 analysis <- append(analysis, line_with_significance_level)

 analysis <- paste(analysis, collapse = "\n")
 class(analysis) <- "analysis_of_variance"
 return(analysis)
}
