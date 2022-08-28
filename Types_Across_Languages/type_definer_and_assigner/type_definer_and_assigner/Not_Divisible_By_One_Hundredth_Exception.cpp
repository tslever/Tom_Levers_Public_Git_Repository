#include <exception>

#include "Not_Divisible_By_One_Hundredth_Exception.h"


namespace type_definer_and_assigner {


	Not_Divisible_By_One_Hundredth_Exception::Not_Divisible_By_One_Hundredth_Exception(
		const char* message_to_use) {
		this->message = message_to_use;
	}

	const char* Not_Divisible_By_One_Hundredth_Exception::what() const throw() {
		return this->message;
	};

};