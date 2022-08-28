#ifndef TEMPERATURE_VALUE_GREATER_THAN_UPPER_LIMIT_EXCEPTION
#define TEMPERATURE_VALUE_GREATER_THAN_UPPER_LIMIT_EXCEPTION


#include <exception>


namespace type_definer_and_assigner {


	class Temperature_Value_Greater_Than_Upper_Limit_Exception : public std::exception {

	private:
		const char* message;

	public:
		Temperature_Value_Greater_Than_Upper_Limit_Exception(const char* message_to_use);

		virtual const char* what() const throw();

	};

};


#endif