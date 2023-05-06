package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForClearBuffer encapsulates a version of the edit method by which the line editor clears all lines in
 * its buffer of strings.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForClearBuffer extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to clear all lines in its buffer of strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		LineEditor.bufferOfStrings.empty();
		
	}
	
}