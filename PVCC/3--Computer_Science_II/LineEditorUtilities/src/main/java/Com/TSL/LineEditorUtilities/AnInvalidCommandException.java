package Com.TSL.LineEditorUtilities;


/**
 * AnInvalidCommandException represents the structure for an exception that occurs when an input manager receives an
 * invalid command.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnInvalidCommandException extends Exception {

	
	/**
	 * AnInvalidCommandException(String message) is the one-parameter constructor for AnInvalidCommandException, which
	 * passes a provided message to Exception's one-parameter constructor.
	 * 
	 * @param message
	 */
	
	public AnInvalidCommandException(String message) {
		super(message);
	}
	
}