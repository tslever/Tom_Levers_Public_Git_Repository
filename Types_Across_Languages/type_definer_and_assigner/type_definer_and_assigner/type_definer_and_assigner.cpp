#include <iostream>

#include "A_Temperature.h"
#include "A_Temperature_Scale.h"


using namespace type_definer_and_assigner;


int main()
{
	A_Temperature my_temperature = A_Temperature();

	//my_temperature = 100.0;
	// Visual Studio 2019 throws compile-time error 'no operator "=" matches these operands'.

	try {
		my_temperature.Takes_A_Value_Of(100.0);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	// Valid

	try {
		my_temperature.Takes_A_Value_Of(-20000.0);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	// Program throws runtime "Exception: Temperature value less than lower limit.".

	try {
		my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale::CELSIUS, 0.0);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	// Valid

	try {
		my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale::FAHRENHEIT, 0.0);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	// Valid

	/*try {
		my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Unit(Kelvin), 0.0);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}*/
	// Visual Studio throws compile-time error 'identifier "Kelvin" is undefined'.

	try {
		my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale::CELSIUS, -302.00);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	// Program throws runtime "Exception: Temperature value less than lower limit.".


	return 0;

}
