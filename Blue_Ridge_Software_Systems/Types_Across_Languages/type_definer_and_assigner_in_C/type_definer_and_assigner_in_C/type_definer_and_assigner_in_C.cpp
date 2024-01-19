#include "An_Error_Code.h"
#include "A_Temperature.h"


int main()
{
    A_Temperature my_temperature = A_New_Temperature();

    //the_temperature = 100.0;
    // Visual Studio 2019 throws compile-time error 'no operator "=" matches these operands'.

    Evaluate(Takes_A_Value_Of(&my_temperature, 100.0)); // Valid

    Evaluate(Takes_A_Value_Of(&my_temperature, -20000.0));
    // Program throws runtime "Exception: Temperature value less than lower limit.".

    Evaluate(Takes_A_Unit_And_A_Value_Of(&my_temperature, CELSIUS, 0.0));

    Evaluate(Takes_A_Unit_And_A_Value_Of(&my_temperature, FAHRENHEIT, 0.0));

    //Evaluate(Takes_A_Unit_And_A_Value_Of(&my_temperature, KELVIN, 0.0));
    // Visual Studio 2019 throws compile-time error 'identifier "KELVIN" is undefined'.

    Evaluate(Takes_A_Unit_And_A_Value_Of(&my_temperature, CELSIUS, -302.00));
    // Program throws runtime "Exception: Temperature value less than lower limit.".

    return 0;

}
