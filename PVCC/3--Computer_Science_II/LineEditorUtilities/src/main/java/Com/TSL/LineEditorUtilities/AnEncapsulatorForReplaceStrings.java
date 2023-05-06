package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForReplaceStrings encapsulates a version of the edit method by which the line editor replaces all
 * occurrences in each line in the line editor's buffer of strings of a first provided string with a second provided
 * string.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForReplaceStrings extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to replace all occurrences in each line in the line editor's buffer of strings of a
	 * first provided string with a second provided string.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		LineEditor.bufferOfStrings.replace(theArrayOfArguments[0], theArrayOfArguments[1]);
		
	}
	
}