determine_whether_linear_model_is_homoscedastic <- function(linear_model) {
    breusch_pagan_test_result <- bptest(linear_model)
    p_value <- breusch_pagan_test_result$p.value
    significance_level <- 0.05
    if (p_value >= significance_level) {
        return(TRUE)
    } else {
        return(FALSE)
    }
}
