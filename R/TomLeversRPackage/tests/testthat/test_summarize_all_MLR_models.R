test_that("summarize_all_MLR_models works",  {
    library(monomvn)
    ln <- function(response) { return(log(response, base = exp(1))) }
    serialized_ln <- deparse(ln, control = "all")
    number_of_possible_MLR_models <- 15
    names_of_variables <- c("y", "x1", "x2", "x3", "x4")
    column <- rep(c(serialized_ln), times = number_of_possible_MLR_models)
    functions_to_apply <- data.frame(rings = column)
    for (name_of_variable in names_of_variables) {
        functions_to_apply[name_of_variable] <- column
    }
    maximum_number_of_MLR_models_for_one_number_of_predictors <- 6
    summary_of_all_MLR_models <- summarize_all_MLR_models(cement, "y", maximum_number_of_MLR_models_for_one_number_of_predictors, functions_to_apply)
    print(summary_of_all_MLR_models$data_frame)
})
