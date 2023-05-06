package Com.TSL.LineEditorUtilities;


/**
 * AnInsertsStringException represents the structure for an exception that occurs when a buffer of strings receives a
 * null reference to a string or a reference to an empty string.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnInsertsStringException extends RuntimeException { // TODO: Change to Exception after testing.

	
	/**
	 * AnInsertsStringException(String message) is the one-parameter constructor for AnInsertsStringException, which
	 * passes a provided message to Exception's one-parameter constructor.
	 * 
	 * @param message
	 */
	
	public AnInsertsStringException(String message) {
		super(message);
	}
	
}