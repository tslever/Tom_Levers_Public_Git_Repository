#include "A_Temperature.h"
#include "A_Temperature_Scale.h"
#include "A_Temperature_Unit.h"


namespace type_definer_and_assigner {


	A_Temperature::A_Temperature() {
		this->the_temperature_scale = A_Temperature_Scale::CELSIUS;
		this->the_temperature_value = -273.15;
	};


	void A_Temperature::Takes_A_Unit_Of(A_Temperature_Scale the_temperature_scale_to_use) {
		this->the_temperature_scale = the_temperature_scale_to_use;
	}


	void A_Temperature::Check(double the_temperature_value_to_check) {

		if (round(the_temperature_value_to_check * 100.0) / 100.0 !=
			the_temperature_value_to_check) {
			throw Not_Divisible_By_One_Hundredth_Exception(
				"Exception: Temperature value not divisible by one hundredth.");
		}

		if (this->the_temperature_scale == A_Temperature_Scale::CELSIUS) {

			if (the_temperature_value_to_check < -273.15) {
				throw Temperature_Value_Less_Than_Lower_Limit_Exception(
					"Exception: Temperature value less than lower limit.");
			};

			if (the_temperature_value_to_check > 10000.0) {
				throw Temperature_Value_Greater_Than_Upper_Limit_Exception(
					"Exception: Temperature value greater than upper limit.");
			}

		}

		if (this->the_temperature_scale == A_Temperature_Scale::FAHRENHEIT) {

			if (the_temperature_value_to_check < -459.67) {
				throw Temperature_Value_Less_Than_Lower_Limit_Exception(
					"Exception: Temperature value less than lower limit.");
			};

			if (the_temperature_value_to_check > 10000.0) {
				throw Temperature_Value_Greater_Than_Upper_Limit_Exception(
					"Exception: Temperature value greater than upper limit.");
			}

		}

	};


	void A_Temperature::Takes_A_Value_Of(double the_temperature_value_to_use) {

		try {
			Check(the_temperature_value_to_use);
		}
		catch (Not_Divisible_By_One_Hundredth_Exception& e) {
			throw Not_Divisible_By_One_Hundredth_Exception(e.what());
		}
		catch (Temperature_Value_Less_Than_Lower_Limit_Exception& e) {
			throw Temperature_Value_Less_Than_Lower_Limit_Exception(e.what());
		}
		catch (Temperature_Value_Greater_Than_Upper_Limit_Exception& e) {
			throw Temperature_Value_Greater_Than_Upper_Limit_Exception(e.what());
		}

		this->the_temperature_value = the_temperature_value_to_use;

	};


	void A_Temperature::Takes_A_Unit_And_A_Value_Of(
		A_Temperature_Scale the_temperature_scale_to_use, double the_temperature_value_to_use) {

		this->the_temperature_scale = the_temperature_scale_to_use;

		try {
			Check(the_temperature_value_to_use);
		}
		catch (Not_Divisible_By_One_Hundredth_Exception& e) {
			throw Not_Divisible_By_One_Hundredth_Exception(e.what());
		}
		catch (Temperature_Value_Less_Than_Lower_Limit_Exception& e) {
			throw Temperature_Value_Less_Than_Lower_Limit_Exception(e.what());
		}
		catch (Temperature_Value_Greater_Than_Upper_Limit_Exception& e) {
			throw Temperature_Value_Greater_Than_Upper_Limit_Exception(e.what());
		}

		this->the_temperature_value = the_temperature_value_to_use;

	};


};