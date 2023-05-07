package com.tsl.more_operations_with_arrays.input_output_utilities;


/**
 * AProposedNumberOfElementsIsNegativeExceptionException represents the structure for an exception that occurs when a read integer
 * is not positive.
 * @author Tom Lever
 * @version 1.0
 * @since 05/18/21
 */
class AProposedNumberOfElementsIsNegativeException extends Exception {

	
	/**
	 * AProposedNumberOfElementsIsNegativeExceptionException() is a conventional zero-argument constructor for
	 * AProposedNumberOfElementsIsNegativeExceptionException, which calls Exception's zero-argument constructor.
	 */
	AProposedNumberOfElementsIsNegativeException() {
		super();
	}
	
	
	/**
	 * AProposedNumberOfElementsIsNegativeExceptionException(String message) is a one-argument constructor for
	 * AProposedNumberOfElementsIsNegativeExceptionException, which passes an error message to Exception's one-argument
	 * constructor with a message argument.
	 * @param message
	 */
	public AProposedNumberOfElementsIsNegativeException(String message) {
		super(message);
	}
	
}