package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForCountWords encapsulates a version of the edit method by which the line editor displays the number of
 * words in its buffer of strings.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForCountWords extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to display the number of words in its buffer of strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		System.out.print(
			"The number of words in the line editor's buffer of strings is " + LineEditor.bufferOfStrings.words() +
			".\n\n"
		);
		
	}
	
}