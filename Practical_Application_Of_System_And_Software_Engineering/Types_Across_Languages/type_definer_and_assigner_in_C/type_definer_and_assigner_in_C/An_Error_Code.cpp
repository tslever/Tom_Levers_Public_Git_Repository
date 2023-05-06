#include <stdio.h>

#include "An_Error_Code.h"


void Evaluate(An_Error_Code the_error_code) {

    if (the_error_code == NO_EXCEPTION) {
        fprintf(stderr, "No Exception.\n");
    }

    else if (the_error_code == POINTER_TO_THE_TEMPERATURE_IS_NULL_EXCEPTION) {
        fprintf(stderr, "Exception: Pointer to the temperature is null.");
    }

    else if (the_error_code == NOT_DIVISIBLE_BY_ONE_HUNDREDTH_EXCEPTION) {
        fprintf(stderr, "Exception: Temperature value not divisible by one hundredth.\n");
    }

    else if (the_error_code == TEMPERATURE_VALUE_LESS_THAN_LOWER_LIMIT_EXCEPTION) {
        fprintf(stderr, "Exception: Temperature value less than lower limit.\n");
    }

    else if (the_error_code == TEMPERATURE_VALUE_GREATER_THAN_UPPER_LIMIT_EXCEPTION) {
        fprintf(stderr, "Exception: Temperature value greater than upper limit.\n");
    }

    else {
        fprintf(stderr, "Unknown exception.\n");
    }

}