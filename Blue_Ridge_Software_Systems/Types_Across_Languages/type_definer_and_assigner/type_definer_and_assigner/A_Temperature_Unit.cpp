#include "A_Temperature_Scale.h"
#include "A_Temperature_Unit.h"


namespace type_definer_and_assigner {

	A_Temperature_Unit::A_Temperature_Unit() {

	};

	A_Temperature_Unit::A_Temperature_Unit(A_Temperature_Scale the_temperature_scale_to_use) {
		this->the_temperature_scale = the_temperature_scale_to_use;
	};

	A_Temperature_Scale A_Temperature_Unit::The_Temperature_Scale() {
		return this->the_temperature_scale;
	};

};