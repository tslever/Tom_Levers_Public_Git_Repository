package Com.TSL.RecursionWithArrays;


/** *************************************************************************************************************
* IllegalArgumentException represents the structure for an exception that occurs if a program requests a negative
* number of copies of a string.
* 
* @author Tom Lever
* @version 1.0
* @since 05/18/21
************************************************************************************************************** */

class IllegalArgumentException extends Exception {

	
	/** ---------------------------------------------------------------------------------------------------------------
	 * IllegalArgumentException() is a conventional zero-argument constructor for IllegalArgumentException, which calls
	 * Exception's zero-argument constructor.
	 --------------------------------------------------------------------------------------------------------------- */
	
	IllegalArgumentException() {
		super();
	}
	
	
	/** ----------------------------------------------------------------------------------------------------------------
	 * IllegalArgumentException(String message) is a one-argument constructor for IllegalArgumentException, which passes
	 * an error message to Exception's one-argument constructor.
	 * 
	 * @param message
	 ---------------------------------------------------------------------------------------------------------------- */
	
	IllegalArgumentException(String message) {
		super(message);
	}
	
}