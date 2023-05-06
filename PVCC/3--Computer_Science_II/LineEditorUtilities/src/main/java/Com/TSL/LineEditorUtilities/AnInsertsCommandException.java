package Com.TSL.LineEditorUtilities;


/**
 * AnInsertsCommandException represents the structure for an exception that occurs when a command menu receives a
 * null reference to a command or a reference to a command that is already in the command menu.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnInsertsCommandException extends Exception {

	
	/**
	 * AnInsertsCommandException(String message) is the one-parameter constructor for AnInsertsCommandException, which
	 * passes a provided message to Exception's one-parameter constructor.
	 * 
	 * @param message
	 */
	
	public AnInsertsCommandException(String message) {
		super(message);
	}
	
}