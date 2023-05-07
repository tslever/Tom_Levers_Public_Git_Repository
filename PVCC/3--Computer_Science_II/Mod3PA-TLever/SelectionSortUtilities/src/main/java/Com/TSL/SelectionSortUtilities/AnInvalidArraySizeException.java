package Com.TSL.SelectionSortUtilities;


/** ********************************************************************************************************************
* AnInvalidArraySizeException represents the structure for an exception that occurs if the integer representing an array
* size is negative.
* 
* @author Tom Lever
* @version 1.0
* @since 05/18/21
********************************************************************************************************************* */

class AnInvalidArraySizeException extends Exception {

	
	/** ---------------------------------------------------------------------------------------------------------------
	 * AnInvalidArraySizeException() is a conventional zero-argument constructor for AnInvalidArraySizeException, which
	 * calls Exception's zero-argument constructor.
	 --------------------------------------------------------------------------------------------------------------- */
	
	AnInvalidArraySizeException() {
		super();
	}
	
	
	/** ---------------------------------------------------------------------------------------------------------------
	 * AnInvalidArraySizeException(String message) is a one-argument constructor for AnInvalidArraySizeException, which
	 * passes an error message to Exception's one-argument constructor with
	 * a message argument.
	 * 
	 * @param message
	 -------------------------------------------------------------------------------------------------------------- */
	
	AnInvalidArraySizeException(String message) {
		super(message);
	}
	
}