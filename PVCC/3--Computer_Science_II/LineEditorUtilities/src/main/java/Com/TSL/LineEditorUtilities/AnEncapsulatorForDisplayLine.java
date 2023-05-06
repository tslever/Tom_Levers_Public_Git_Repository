package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForDisplayLine encapsulates a version of the edit method by which the line editor displays a line with
 * a provided index.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForDisplayLine extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to display a line with a provided index.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		try {
			LineEditor.bufferOfStrings.line(Integer.parseInt(theArrayOfArguments[0]));
		}
		catch (NumberFormatException theNumberFormatException) {
			System.out.print(
				"The line editor found that the string representing an index of a line was not an integer.\n\n"
			);
		}
		
	}
	
}