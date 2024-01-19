#ifndef A_TEMPERATURE
#define A_TEMPERATURE


#include <math.h>
#include <string>
#include <typeinfo>

#include "A_Temperature_Unit.h"
#include "Not_Divisible_By_One_Hundredth_Exception.h"
#include "Temperature_Value_Greater_Than_Upper_Limit_Exception.h"
#include "Temperature_Value_Less_Than_Lower_Limit_Exception.h"


namespace type_definer_and_assigner {


	class A_Temperature {

	private:
		A_Temperature_Scale the_temperature_scale;
		double the_temperature_value;
	

	public:
		A_Temperature();

		void Takes_A_Unit_Of(A_Temperature_Scale the_temperature_scale_to_use);

		void Check(double the_temperature_value_to_check);

		void Takes_A_Value_Of(double the_temperature_value_to_use);

		void Takes_A_Unit_And_A_Value_Of(
			A_Temperature_Scale the_temperature_scale_to_use, double the_temperature_value_to_use);

	};

}


#endif