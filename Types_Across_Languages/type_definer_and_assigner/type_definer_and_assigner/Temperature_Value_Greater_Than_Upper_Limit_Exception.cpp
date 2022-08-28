#include <exception>

#include "Temperature_Value_Greater_Than_Upper_Limit_Exception.h"


namespace type_definer_and_assigner {


	Temperature_Value_Greater_Than_Upper_Limit_Exception::Temperature_Value_Greater_Than_Upper_Limit_Exception(
		const char* message_to_use) {
		this->message = message_to_use;
	}

	const char* Temperature_Value_Greater_Than_Upper_Limit_Exception::what() const throw() {
		return this->message;
	};

};