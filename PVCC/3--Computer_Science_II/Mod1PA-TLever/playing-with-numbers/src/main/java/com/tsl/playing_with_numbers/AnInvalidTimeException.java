package com.tsl.playing_with_numbers;


/**
 * AnInvalidTimeException represents the structure for an exception that occurs when a proposed time is outside of the
 * interval 1 to 6.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class AnInvalidTimeException extends Exception {

	/**
	 * AnInvalidTimeException() is a conventional zero-argument constructor for AnInvalidTimeException, which calls
	 * Exception's zero-argument constructor.
	 */
	AnInvalidTimeException() {
		super();
	}
	
	
	/**
	 * AnInvalidTimeException(String message) is a one-argument constructor for AnInvalidTimeException, which passes
	 * an error message to Exception's one-argument constructor with a message argument.
	 * @param message
	 */
	public AnInvalidTimeException(String message) {
		super(message);
	}
	
}