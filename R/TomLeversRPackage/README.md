# TomLeversRPackage
Contains Tom Lever's R functions.

To rebuild this package, in RStudio's console, run

`install.packages("devtools")`

`setwd("C:/Users/Tom/Documents/Tom_Levers_Public_Git_Repository/R/TomLeversRPackage")`

`devtools::document()`

To install this package, in RStudio's console, run

`detach("package:TomLeversRPackage", unload = TRUE)`

`install.packages("C:/Users/Tom/Documents/Tom_Levers_Git_Repository/R/TomLeversRPackage", repos = NULL, type="source")`

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
