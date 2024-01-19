#include <math.h>

#include "A_Temperature.h"
#include "An_Error_Code.h"


void Takes_A_Unit_Of(
	A_Temperature* the_temperature_to_use, A_Temperature_Scale the_temperature_scale_to_use) {

	the_temperature_to_use->the_temperature_scale = the_temperature_scale_to_use;

}


An_Error_Code Check(A_Temperature* the_temperature, double the_temperature_value) {

	if (!the_temperature) {
		return POINTER_TO_THE_TEMPERATURE_IS_NULL_EXCEPTION;
	}

	if (round(the_temperature_value * 100.0) / 100.0 != the_temperature_value) {
		return NOT_DIVISIBLE_BY_ONE_HUNDREDTH_EXCEPTION;
	}

	if (the_temperature->the_temperature_scale == CELSIUS) {
		
		if (the_temperature_value < -273.15) {
			return TEMPERATURE_VALUE_LESS_THAN_LOWER_LIMIT_EXCEPTION;
		}

		if (the_temperature_value > 10000.0) {
			return TEMPERATURE_VALUE_GREATER_THAN_UPPER_LIMIT_EXCEPTION;
		}

	}

	if (the_temperature->the_temperature_scale == FAHRENHEIT) {

		if (the_temperature_value < -459.67) {
			return TEMPERATURE_VALUE_LESS_THAN_LOWER_LIMIT_EXCEPTION;
		}

		if (the_temperature_value > 10000.0) {
			return TEMPERATURE_VALUE_GREATER_THAN_UPPER_LIMIT_EXCEPTION;
		}

	}

	return NO_EXCEPTION;

}


An_Error_Code Takes_A_Value_Of(
	A_Temperature* the_temperature, double the_temperature_value) {

	An_Error_Code the_error_code = Check(the_temperature, the_temperature_value);
	if (the_error_code != NO_EXCEPTION) {
		return the_error_code;
	}

	the_temperature->the_temperature_value = the_temperature_value;

	return NO_EXCEPTION;

}


An_Error_Code Takes_A_Unit_And_A_Value_Of(
	A_Temperature* the_temperature,
	A_Temperature_Scale the_temperature_scale_to_use,
	double the_temperature_value_to_use
) {

	the_temperature->the_temperature_scale = the_temperature_scale_to_use;

	An_Error_Code the_error_code = Check(the_temperature, the_temperature_value_to_use);
	if (the_error_code != NO_EXCEPTION) {
		return the_error_code;
	}

	the_temperature->the_temperature_value = the_temperature_value_to_use;

	return NO_EXCEPTION;

}


A_Temperature A_New_Temperature() {

	A_Temperature the_temperature;

	Takes_A_Unit_And_A_Value_Of(&the_temperature, CELSIUS, -273.15);

	return the_temperature;

}