package Com.TSL.LineEditorUtilities;


/**
 * ACommandMenuHasBeenSetUpException represents the structure for an exception that occurs when the command-menu
 * generator tries to set up a command menu that has been already set up.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class ACommandMenuHasBeenSetUpException extends Exception {

	
	/**
	 * ACommandMenuHasBeenSetUpException(String message) is the one-parameter constructor for
	 * ACommandMenuHasBeenSetUpException, which passes a provided message to Exception's one-parameter constructor.
	 * 
	 * @param message
	 */
	
	public ACommandMenuHasBeenSetUpException(String message) {
		super(message);
	}
	
}