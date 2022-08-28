#ifndef A_TEMPERATURE
#define A_TEMPERATURE


#include "An_Error_Code.h"
#include "A_Temperature_Scale.h"


typedef struct A_Temperature {

    A_Temperature_Scale the_temperature_scale;
    double the_temperature_value;

} A_Temperature;


void Takes_A_Unit_Of(A_Temperature* the_temperature, A_Temperature_Scale the_temperature_scale);

An_Error_Code Takes_A_Value_Of(A_Temperature* the_temperature, double the_temperature_value);

An_Error_Code Takes_A_Unit_And_A_Value_Of(
    A_Temperature* the_temperature,
    A_Temperature_Scale the_temperature_scale_to_use,
    double the_temperature_value_to_use);

A_Temperature A_New_Temperature();


#endif