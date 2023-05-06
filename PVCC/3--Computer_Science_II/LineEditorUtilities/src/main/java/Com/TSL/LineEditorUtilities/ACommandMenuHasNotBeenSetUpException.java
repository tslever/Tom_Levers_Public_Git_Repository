package Com.TSL.LineEditorUtilities;


/**
 * ACommandMenuHasNotBeenSetUpException represents the structure for an exception that occurs when the command-menu
 * generator tries to provide a command menu that has not been set up.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class ACommandMenuHasNotBeenSetUpException extends Exception {

	
	/**
	 * ACommandMenuHasNotBeenSetUpException(String message) is the one-parameter constructor for
	 * ACommandMenuHasNotBeenSetUpException, which passes a provided message to Exception's one-parameter constructor.
	 * 
	 * @param message
	 */
	
	public ACommandMenuHasNotBeenSetUpException(String message) {
		super(message);
	}
	
}