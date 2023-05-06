#ifndef A_TEMPERATURE_UNIT
#define A_TEMPERATURE_UNIT


#include "A_Temperature_Scale.h"


namespace type_definer_and_assigner {

	class A_Temperature_Unit {

	private:
		A_Temperature_Scale the_temperature_scale;

	public:
		A_Temperature_Unit();

		A_Temperature_Unit(A_Temperature_Scale the_temperature_scale_to_use);

		A_Temperature_Scale The_Temperature_Scale();

	};

}


#endif