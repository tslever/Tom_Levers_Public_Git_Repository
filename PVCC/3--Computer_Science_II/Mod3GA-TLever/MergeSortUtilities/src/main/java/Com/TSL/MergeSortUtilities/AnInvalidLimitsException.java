package Com.TSL.MergeSortUtilities;


/** *****************************************************************************************************************
* AnInvalidLimitsException represents the structure for an exception that occurs if the a lower limit is greater than
* an upper limit.
* 
* @author Tom Lever
* @version 1.0
* @since 05/18/21
****************************************************************************************************************** */

class AnInvalidLimitsException extends Exception {

	
	/** ---------------------------------------------------------------------------------------------------------------
	 * AnInvalidLimitsException() is a conventional zero-argument constructor for AnInvalidLimitsException, which calls
	 * Exception's zero-argument constructor.
	 --------------------------------------------------------------------------------------------------------------- */
	
	AnInvalidLimitsException() {
		super();
	}
	
	
	/** ----------------------------------------------------------------------------------------------------------------
	 * AnInvalidLimitsException(String message) is a one-argument constructor for AnInvalidLimitsException, which passes
	 * an error message to Exception's one-argument constructor with a message argument.
	 * 
	 * @param message
	 ---------------------------------------------------------------------------------------------------------------- */
	
	AnInvalidLimitsException(String message) {
		super(message);
	}
	
}