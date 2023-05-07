package com.tsl.playing_with_numbers;


/**
* ANoMinimumIntegerExistsException represents the structure for an exception that occurs when no minimum integer exists
* in a stack of integers.
* @author Tom Lever
* @version 1.0
* @since 05/28/21
*/
class ANoMinimumIntegerExistsException extends Exception {

	
	/**
	 * ANoMinimumIntegerExistsException() is a conventional zero-parameter constructor for
	 * ANoMinimumIntegerExistsException, which calls Exception's zero-parameter constructor.
	 */
	protected ANoMinimumIntegerExistsException() {
		super();
	}
	
	
	/**
	 * ANoMinimumIntegerExistsException(String message) is a one-parameter constructor for
	 * ANoMinimumIntegerExistsException, which passes an error message to Exception's one-parameter constructor with a
	 * message parameter.
	 * @param message
	 */
	protected ANoMinimumIntegerExistsException(String message) {
		super(message);
	}
	
}