package Com.TSL.RecursionWithArrays;


/** ********************************************************************************************************************
* ANoMinimumExistsException represents the structure for an exception that occurs if a program requests the minimum from
* an empty array.
* 
* @author Tom Lever
* @version 1.0
* @since 05/18/21
********************************************************************************************************************* */

class ANoMinimumExistsException extends Exception {

	
	/** -----------------------------------------------------------------------------------------------------------------
	 * ANoMinimumExistsException() is a conventional zero-argument constructor for ANoMinimumExistsException, which calls
	 * Exception's zero-argument constructor.
	 ----------------------------------------------------------------------------------------------------------------- */
	
	ANoMinimumExistsException() {
		super();
	}
	
	
	/** -----------------------------------------------------------------------------------------------------------
	 * ANoMinimumExistsException(String message) is a one-argument constructor for ANoMinimumExistsException, which
	 * passes an error message to Exception's one-argument constructor with a message argument.
	 * 
	 * @param message
	 ----------------------------------------------------------------------------------------------------------- */
	
	ANoMinimumExistsException(String message) {
		super(message);
	}
	
}