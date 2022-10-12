# TomLeversRPackage
Contains Tom Lever's R functions

1. analyze_variance
2. calculate_mode
3. calculate_percentile
4. construct_confidence_interval_for_difference_between_two_population_means
5. construct_confidence_interval_for_population_mean
6. hello
7. summarize_linear_model
8. test_null_hypothesis_involving_proportion

To rebuild this package, open the TomLeversRPackage project in RStudio and run devtools::document() in the RStudio console.

To install this package, run in the RStudio console
install.packages("~\\Documents\\Tom_Levers_Git_Repository\\TomLeversRPackage", repo = NULL, type="source")

Regarding creating this package
https://www.r-bloggers.com/2020/07/how-to-write-your-own-r-package-and-publish-it-on-cran/

Regarding creating tests
https://r-pkgs.org/testing-basics.html

Per "Writing R Extensions" at https://cran.r-project.org/doc/manuals/R-exts.html#The-DESCRIPTION-file,
"The mandatory ‘Package’ field gives the name of the package. This should contain only (ASCII) letters, numbers and dot, have at least two characters and start with a letter and not end in a dot."

Per "Files" at https://style.tidyverse.org/files.html,
"Avoid using special characters in file names - stick with numbers, letters, -, and _.
Good fit_models.R"

Per "Syntax",
"Variable and function names should use only lowercase letters, numbers, and \_. Use underscores (\_) (so called snake case) to separate words within a name.
Good: day_one"
