#include <exception>

#include "Temperature_Value_Less_Than_Lower_Limit_Exception.h"


namespace type_definer_and_assigner {

	Temperature_Value_Less_Than_Lower_Limit_Exception::Temperature_Value_Less_Than_Lower_Limit_Exception(const char* message_to_use) {
		this->message = message_to_use;
	}

	const char* Temperature_Value_Less_Than_Lower_Limit_Exception::what() const throw() {
		return this->message;
	};

};