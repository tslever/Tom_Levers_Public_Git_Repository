package com.tsl.polynomials;


/**
 * AnInvalidDegreeException represents the structure for an exception that occurs when a passed degree is invalid.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
public class AnInvalidDegreeException extends Exception {

	/**
	 * AnInvalidDegreeException() is a conventional zero-argument constructor for AnInvalidDegreeException, which
	 * calls Exception's zero-argument constructor.
	 */
	AnInvalidDegreeException() {
		super();
	}
	
	
	/**
	 * AnInvalidDegreeException(String message) is a one-argument constructor for AnInvalidDegreeException, which
	 * passes an error message to Exception's one-argument constructor with a message argument.
	 * @param message
	 */
	public AnInvalidDegreeException(String message) {
		super(message);
	}
	
}