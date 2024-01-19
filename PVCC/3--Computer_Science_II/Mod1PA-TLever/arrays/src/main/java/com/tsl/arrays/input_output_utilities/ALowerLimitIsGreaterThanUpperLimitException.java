package com.tsl.more_operations_with_arrays.input_output_utilities;


/**
 * ALowerLimitIsGreaterThanUpperLimitException represents the structure for an exception that occurs when a lower limit
 * is greater than an upper limit.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class ALowerLimitIsGreaterThanUpperLimitException extends Exception {

	/**
	 * ALowerLimitIsGreaterThanUpperLimitException() is a conventional zero-argument constructor for
	 * ALowerLimitIsGreaterThanUpperLimitException, which calls Exception's zero-argument constructor.
	 */
	ALowerLimitIsGreaterThanUpperLimitException() {
		super();
	}
	
	
	/**
	 * ALowerLimitIsGreaterThanUpperLimitException(String message) is a one-argument constructor for
	 * ALowerLimitIsGreaterThanUpperLimitException, which passes an error message to Exception's one-argument
	 * constructor with a message argument.
	 * @param message
	 */
	public ALowerLimitIsGreaterThanUpperLimitException(String message) {
		super(message);
	}
	
}