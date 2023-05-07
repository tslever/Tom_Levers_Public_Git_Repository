package com.tsl.playing_with_numbers;


/**
* ANoAverageExistsException represents the structure for an exception that occurs when no average exists for a stack of
* integers.
* @author Tom Lever
* @version 1.0
* @since 05/28/21
*/
class ANoAverageExistsException extends Exception {

	
	/**
	 * ANoAverageExistsException() is a conventional zero-parameter constructor for ANoAverageExistsException, which
	 * calls Exception's zero-parameter constructor.
	 */
	protected ANoAverageExistsException() {
		super();
	}
	
	
	/**
	 * ANoAverageExistsException(String message) is a one-parameter constructor for ANoAverageExistsException, which
	 * passes an error message to Exception's one-parameter constructor with a message parameter.
	 * @param message
	 */
	protected ANoAverageExistsException(String message) {
		super(message);
	}
	
}