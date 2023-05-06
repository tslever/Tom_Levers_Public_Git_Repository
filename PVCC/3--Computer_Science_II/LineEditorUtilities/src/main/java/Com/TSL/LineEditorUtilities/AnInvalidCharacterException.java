package Com.TSL.LineEditorUtilities;


/**
 * AnInvalidCommandException represents the structure for an exception that occurs when an input manager reads an
 * invalid character from a file.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnInvalidCharacterException extends RuntimeException { // TODO: Change to Exception after testing.

	
	/**
	 * AnInvalidCharacterException(String message) is the one-parameter constructor for AnInvalidCharacterException,
	 * which passes a provided message to Exception's one-parameter constructor.
	 * 
	 * @param message
	 */
	
	public AnInvalidCharacterException(String message) {
		super(message);
	}
	
}