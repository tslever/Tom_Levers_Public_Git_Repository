package com.tsl.grading_exams;


/**
 * AnIntegerOverflowException represents the structure for an exception that occurs when an arithmetic operation
 * would cause an integer to be greater than the maximum integer or less than the minimum integer.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class ADuplicateStudentNameException extends Exception {

	/**
	 * AnIntegerOverflowException() is a conventional zero-argument constructor for AnIntegerOverflowException, which
	 * calls Exception's zero-argument constructor.
	 */
	protected ADuplicateStudentNameException() {
		super();
	}
	
	
	/**
	 * AnIntegerOverflowException(String message) is a one-argument constructor for AnIntegerOverflowException, which
	 * passes an error message to Exception's one-argument constructor with a message argument.
	 * @param message
	 */
	protected ADuplicateStudentNameException(String message) {
		super(message);
	}
	
}