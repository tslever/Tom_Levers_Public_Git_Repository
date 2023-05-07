package com.tsl.polynomials;


/**
 * AnInvalidArrayOfCoefficientsGivenDegreeException represents the structure for an exception that occurs when a passed
 * array of coefficients for a polynomial is invalid given a passed degree.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
public class AnInvalidArrayOfCoefficientsGivenDegreeException extends Exception {

	/**
	 * AnInvalidArrayOfCoefficientsGivenDegreeException() is a conventional zero-argument constructor for
	 * AnInvalidArrayOfCoefficientsGivenDegreeException, which calls Exception's zero-argument constructor.
	 */
	AnInvalidArrayOfCoefficientsGivenDegreeException() {
		super();
	}
	
	
	/**
	 * AnInvalidArrayOfCoefficientsGivenDegreeException(String message) is a one-argument constructor for
	 * AnInvalidArrayOfCoefficientsGivenDegreeException, which passes an error message to Exception's one-argument
	 * constructor with a message argument.
	 * @param message
	 */
	public AnInvalidArrayOfCoefficientsGivenDegreeException(String message) {
		super(message);
	}
	
}