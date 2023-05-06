package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForQuit encapsulates a version of the edit method by which the line editor quits.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForQuit extends AnEncapsulatorForEdit {
	
	
	/**
	 * AnErrorCode clarifies the error code associated with successfully quitting.
	 * 
	 * @author Tom Lever
	 * @version 1.0
	 * @since 06/26/21
	 */
	
	private enum AnErrorCode {
		SUCCESS
	}
	
	
	/**
	 * edit allows LineEditor to quit.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		System.out.println("LineEditor is quitting.");
		System.exit(AnErrorCode.SUCCESS.ordinal());
		
	}
	
}