package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForGetNumberOfLines encapsulates a version of the edit method by which the line editor outputs the
 * number of lines in its buffer of strings.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForGetNumberOfLines extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to output the number of lines in its buffer of strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		System.out.println(
			"The number of lines in the line editor's buffer of strings is " + LineEditor.bufferOfStrings.lines() + ".\n"
		);
		
	}
	
}