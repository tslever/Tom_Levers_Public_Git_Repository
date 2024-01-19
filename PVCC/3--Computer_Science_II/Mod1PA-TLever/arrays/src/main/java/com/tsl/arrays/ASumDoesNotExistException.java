package com.tsl.more_operations_with_arrays;


/**
 * ASumDoesNotExistException represents the structure for an exception that occurs when a sum does not exist because
 * there are no summands.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class ASumDoesNotExistException extends Exception {

	/**
	 * ASumDoesNotExistException() is a conventional zero-argument constructor for ASumDoesNotExistException, which calls
	 * Exception's zero-argument constructor.
	 */
	ASumDoesNotExistException() {
		super();
	}
	
	
	/**
	 * ASumDoesNotExistException(String message) is a one-argument constructor for ASumDoesNotExistException, which
	 * passes an error message to Exception's one-argument constructor with a message argument.
	 * @param message
	 */
	public ASumDoesNotExistException(String message) {
		super(message);
	}
	
}
