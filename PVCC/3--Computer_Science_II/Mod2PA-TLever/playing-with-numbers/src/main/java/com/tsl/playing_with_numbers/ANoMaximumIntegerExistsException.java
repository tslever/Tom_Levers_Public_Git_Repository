package com.tsl.playing_with_numbers;


/**
* ANoMaximumIntegerExistsException represents the structure for an exception that occurs when no maximum integer exists
* in a stack of integers.
* @author Tom Lever
* @version 1.0
* @since 05/28/21
*/
class ANoMaximumIntegerExistsException extends Exception {

	
	/**
	 * ANoMaximumIntegerExistsException() is a conventional zero-parameter constructor for
	 * ANoMaximumIntegerExistsException, which calls Exception's zero-parameter constructor.
	 */
	protected ANoMaximumIntegerExistsException() {
		super();
	}
	
	
	/**
	 * ANoMaximumIntegerExistsException(String message) is a one-parameter constructor for
	 * ANoMaximumIntegerExistsException, which passes an error message to Exception's one-parameter constructor with a
	 * message parameter.
	 * @param message
	 */
	protected ANoMaximumIntegerExistsException(String message) {
		super(message);
	}
	
}