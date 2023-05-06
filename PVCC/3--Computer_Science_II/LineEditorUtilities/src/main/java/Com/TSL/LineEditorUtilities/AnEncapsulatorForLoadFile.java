package Com.TSL.LineEditorUtilities;


/**
 * AnEncapsulatorForLoadFile encapsulates a version of the edit method by which the line editor clears its buffer of
 * strings if an append / overwrite option is false, loads the file at the provided path, and appends the lines in that
 * file to the line editor's buffer of strings.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForLoadFile extends AnEncapsulatorForEdit {
	 
	
	/**
	 * edit allows the line editor to clear its buffer of strings if an append / overwrite option is false, load the
	 * file at the provided path, and append the lines in that file to the line editor's buffer of strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		if (!theArrayOfArguments[1].equals("true") &&
			!theArrayOfArguments[1].equals("false")) {
			System.out.print(
				"An input manager received command \"load\" with an invalid append / overwrite option.\n\n"
			);
			return;
		}
		
		LineEditor.bufferOfStrings.load(theArrayOfArguments[0], Boolean.parseBoolean(theArrayOfArguments[1]));
		
	}
	
}
