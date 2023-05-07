package com.tsl.more_operations_with_arrays;


/**
 * AMaximumValueDoesNotExistException represents the structure for an exception that occurs when a maximum value in an
 * array does not exist because the array has zero elements.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class AMaximumValueDoesNotExistException extends Exception {

	/**
	 * AMaximumValueDoesNotExistException() is a conventional zero-argument constructor for
	 * AMaximumValueDoesNotExistException, which calls Exception's zero-argument constructor.
	 */
	AMaximumValueDoesNotExistException() {
		super();
	}
	
	
	/**
	 * AMaximumValueDoesNotExistException(String message) is a one-argument constructor for
	 * AMaximumValueDoesNotExistException, which passes an error message to Exception's one-argument constructor with a
	 * message argument.
	 * @param message
	 */
	public AMaximumValueDoesNotExistException(String message) {
		super(message);
	}
	
}