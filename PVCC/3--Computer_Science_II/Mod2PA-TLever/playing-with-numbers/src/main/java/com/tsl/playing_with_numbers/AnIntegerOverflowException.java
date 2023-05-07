package com.tsl.playing_with_numbers;


/**
* AnIntegerOverflowException represents the structure for an exception that occurs when an arithmetic operation
* would cause an integer to be greater than the maximum integer or less than the minimum integer.
* @author Tom Lever
* @version 1.0
* @since 05/28/21
*/
class AnIntegerOverflowException extends Exception {

	
	/**
	 * AnIntegerOverflowException() is a conventional zero-parameter constructor for AnIntegerOverflowException, which
	 * calls Exception's zero-parameter constructor.
	 */
	protected AnIntegerOverflowException() {
		super();
	}
	
	
	/**
	 * AnIntegerOverflowException(String message) is a one-parameter constructor for AnIntegerOverflowException, which
	 * passes an error message to Exception's one-parameter constructor with a message parameter.
	 * @param message
	 */
	protected AnIntegerOverflowException(String message) {
		super(message);
	}
	
}