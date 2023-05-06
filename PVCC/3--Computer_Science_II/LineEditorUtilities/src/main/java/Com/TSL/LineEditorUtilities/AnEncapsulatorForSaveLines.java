package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForSaveLines encapsulates a version of the edit method by which the line editor saves the lines in
 * its buffer of strings to a file at a provided path.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForSaveLines extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to clear all lines in its buffer of strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		LineEditor.bufferOfStrings.save(theArrayOfArguments[0]);
		
	}
	
}