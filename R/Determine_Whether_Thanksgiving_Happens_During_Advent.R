library(ggplot2)
library(lubridate)

years <- seq(2022, 2122, 1)
days_of_year_of_Thanksgiving <- integer()
days_of_year_of_first_Sunday_of_Advent <- integer()

for (year in years) {

    November_date <- as.Date(paste(year, "-11-01", sep = ""))
    November_date_is_not_Thursday <- TRUE
    while (November_date_is_not_Thursday) {
        November_weekday <- weekdays(November_date)
        if (November_weekday != "Thursday") {
            November_date <- November_date + 1
        } else {
            November_date_is_not_Thursday <- FALSE
            Thanksgiving_date <- November_date + 21
            day_of_year_of_Thanksgiving <- yday(Thanksgiving_date)
            days_of_year_of_Thanksgiving <- append(days_of_year_of_Thanksgiving, day_of_year_of_Thanksgiving)
        }
    }

    #November_date <- as.Date(paste(year, "-11-30", sep = ""))
    #November_date_is_not_Thursday <- TRUE
    #while (November_date_is_not_Thursday) {
    #November_weekday <- weekdays(November_date)
    # if (November_weekday != "Thursday") {
    #  November_date <- November_date - 1
    # } else {
    #  November_date_is_not_Thursday <- FALSE
    #  Thanksgiving_date <- November_date
    #  day_of_year_of_Thanksgiving <- yday(Thanksgiving_date)
    #  days_of_year_of_Thanksgiving <- append(days_of_year_of_Thanksgiving, day_of_year_of_Thanksgiving)
    # }
    #}

    December_date <- as.Date(paste(year, "-12-24", sep = ""))
    December_date_is_not_Sunday <- TRUE
    while (December_date_is_not_Sunday) {
        December_weekday <- weekdays(December_date)
        if (December_weekday != "Sunday") {
            December_date <- December_date - 1
        } else {
            December_date_is_not_Sunday <- FALSE
            first_Sunday_of_Advent_date <- December_date - 21
            day_of_year_of_First_Sunday_of_Advent <- yday(first_Sunday_of_Advent_date)
            days_of_year_of_first_Sunday_of_Advent <- append(days_of_year_of_first_Sunday_of_Advent, day_of_year_of_First_Sunday_of_Advent)
        }
    }

}

the_plot <- ggplot(
 data.frame(
  year = years,
  day_of_year_of_Thanksgiving = days_of_year_of_Thanksgiving,
  day_of_year_of_first_Sunday_of_Advent = days_of_year_of_first_Sunday_of_Advent
 )
) +
 geom_line(aes(x = years, y = day_of_year_of_Thanksgiving), color = "orange") +
 geom_point(aes(x = years, y = day_of_year_of_Thanksgiving), color = "brown") +
 geom_line(aes(x = years, y = day_of_year_of_first_Sunday_of_Advent), color = "green") +
 geom_point(aes(x = years, y = day_of_year_of_first_Sunday_of_Advent), color = "red") +
 labs(
  x = "year",
  y = "day of year",
  title = "Day of Year vs. Year for Thanksgiving and First Day of Advent"
 ) +
 theme(
  plot.title = element_text(hjust = 0.5, size = 11),
  axis.text.x = element_text(angle = 0)
 )

print(the_plot)
