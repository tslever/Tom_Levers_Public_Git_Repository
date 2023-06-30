test_that("summarize_all_MLR_models works",  {
    library(monomvn)
    number_of_possible_MLR_models <- 15
    maximum_number_of_MLR_models_for_one_number_of_predictors <- 6
    identity_function <- function(response) { return(response) }
    serialized_identity_function <- deparse(identity_function, control = "all")
    names_of_variables <- c("y", "x1", "x2", "x3", "x4")

    column <- rep(c(serialized_identity_function), times = number_of_possible_MLR_models)
    functions_to_apply <- data.frame(rings = column)
    for (name_of_variable in names_of_variables) {
        functions_to_apply[name_of_variable] <- column
    }
    summary_of_all_MLR_models <- summarize_all_MLR_models(cement, "y", maximum_number_of_MLR_models_for_one_number_of_predictors, functions_to_apply, Box_Cox_Method_should_be_performed = TRUE)
    #print(summary_of_all_MLR_models$data_frame)
})
