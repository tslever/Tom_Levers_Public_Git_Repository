package com.tsl.playing_with_numbers;


/**
 * AnInvalidNameException represents the structure for an exception that occurs when a proposed name contains other than
 * Unicode letters and spaces.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class AnInvalidNameException extends Exception {

	/**
	 * AnInvalidNameException() is a conventional zero-argument constructor for AnInvalidNameException, which calls
	 * Exception's zero-argument constructor.
	 */
	AnInvalidNameException() {
		super();
	}
	
	
	/**
	 * AnInvalidNameException(String message) is a one-argument constructor for AnInvalidNameException, which passes
	 * an error message to Exception's one-argument constructor with a message argument.
	 * @param message
	 */
	public AnInvalidNameException(String message) {
		super(message);
	}
	
}