#' @title print.hypothesis_test_result
#' @description Adds to the generic function print the hypothesis_test_result method print.hypothesis_test_result
#' @param the_summary The hypothesis-test result to print
#' @examples print(hypothesis_test_result)
#' @export

print.hypothesis_test_result <- function(hypothesis_test_result) {
    cat(hypothesis_test_result)
}
