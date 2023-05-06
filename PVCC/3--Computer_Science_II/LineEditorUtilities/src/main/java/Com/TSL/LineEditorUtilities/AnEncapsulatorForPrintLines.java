package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForPrintLines encapsulates a version of the edit method by which the line editor outputs the lines of
 * its buffer of strings, along with their indices, to the standard output stream.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForPrintLines extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to output the lines of its buffer of strings, along with their indices, to the
	 * standard output stream.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		LineEditor.bufferOfStrings.print();
		
	}
	
}