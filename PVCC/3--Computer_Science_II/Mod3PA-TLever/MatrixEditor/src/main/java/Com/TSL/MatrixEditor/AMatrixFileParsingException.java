package Com.TSL.MatrixEditor;


/** ******************************************************************************************************************
* AMatrixFileParsingException represents the structure for an exception that occurs if the content of a matrix file is
* invalid.
* 
* @author Tom Lever
* @version 1.0
* @since 05/18/21
******************************************************************************************************************* */

class AMatrixFileParsingException extends Exception {

	
	/** ---------------------------------------------------------------------------------------------------------------
	 * AMatrixFileParsingException() is a conventional zero-argument constructor for AMatrixFileParsingException, which
	 * calls Exception's zero-argument constructor.
	 --------------------------------------------------------------------------------------------------------------- */
	
	AMatrixFileParsingException() {
		super();
	}
	
	
	/** ---------------------------------------------------------------------------------------------------------------
	 * AMatrixFileParsingException(String message) is a one-argument constructor for AMatrixFileParsingException, which
	 * passes an error message to Exception's one-argument constructor with a message argument.
	 * 
	 * @param message
	 --------------------------------------------------------------------------------------------------------------- */
	
	AMatrixFileParsingException(String message) {
		super(message);
	}
	
}