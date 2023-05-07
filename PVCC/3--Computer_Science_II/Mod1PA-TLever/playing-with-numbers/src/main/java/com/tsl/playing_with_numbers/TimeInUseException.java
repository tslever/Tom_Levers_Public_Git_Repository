package com.tsl.playing_with_numbers;


/**
 * TimeInUseException represents the structure for an exception that occurs when a proposed time is already in use.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class TimeInUseException extends Exception {

	/**
	 * TimeInUseException() is a conventional zero-argument constructor for TimeInUseException, which calls Exception's
	 * zero-argument constructor.
	 */
	TimeInUseException() {
		super();
	}
	
	
	/**
	 * TimeInUseException(String message) is a one-argument constructor for TimeInUseException, which passes an error
	 * message to Exception's one-argument constructor with a message argument.
	 * @param message
	 */
	public TimeInUseException(String message) {
		super(message);
	}
	
}