package com.tsl.more_operations_with_arrays;


/**
 * ASecondMaximumValueDoesNotExistException represents the structure for an exception that occurs when a second maximum
 * value in an array does not exist because the array has zero or one elements.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class ASecondMaximumValueDoesNotExistException extends Exception {

	/**
	 * ASecondMaximumValueDoesNotExistException() is a conventional zero-argument constructor for
	 * ASecondMaximumValueDoesNotExistException, which calls Exception's zero-argument constructor.
	 */
	ASecondMaximumValueDoesNotExistException() {
		super();
	}
	
	
	/**
	 * ASecondMaximumValueDoesNotExistException(String message) is a one-argument constructor for
	 * ASecondMaximumValueDoesNotExistException, which passes an error message to Exception's one-argument constructor
	 * with a message argument.
	 * @param message
	 */
	public ASecondMaximumValueDoesNotExistException(String message) {
		super(message);
	}
	
}