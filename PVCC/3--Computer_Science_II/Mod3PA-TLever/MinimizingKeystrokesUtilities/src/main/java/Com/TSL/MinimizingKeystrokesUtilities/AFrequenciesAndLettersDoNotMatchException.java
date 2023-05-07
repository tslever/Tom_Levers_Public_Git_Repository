package Com.TSL.MinimizingKeystrokesUtilities;


/** ************************************************************************************************************
* AFrequenciesAndLettersDoNotMatchException represents the structure for an exception that occurs if an array of
* frequencies does not match an array of letters.
* 
* @author Tom Lever
* @version 1.0
* @since 05/18/21
************************************************************************************************************* */

class AFrequenciesAndLettersDoNotMatchException extends Exception {

	
	/** -------------------------------------------------------------------------------------------
	 * AFrequenciesAndLettersDoNotMatchException() is a conventional zero-argument constructor for
	 * AFrequenciesAndLettersDoNotMatchException, which calls Exception's zero-argument constructor.
	 -------------------------------------------------------------------------------------------- */
	
	AFrequenciesAndLettersDoNotMatchException() {
		super();
	}
	
	
	/** ----------------------------------------------------------------------------------------------------------------
	 * AFrequenciesAndLettersDoNotMatchException(String message) is a one-argument constructor for
	 * AFrequenciesAndLettersDoNotMatchException, which passes an error message to Exception's one-argument constructor.
	 * 
	 * @param message
	 --------------------------------------------------------------------------------------------------------------- */
	
	AFrequenciesAndLettersDoNotMatchException(String message) {
		super(message);
	}
	
}
