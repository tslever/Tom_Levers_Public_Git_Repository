package Com.TSL.LineEditorUtilities;


import java.util.Scanner;


/**
 * AnEncapsulatorForAppendLine encapsulates a version of the edit method by which the line editor appends a line to the
 * line editor's buffer of strings.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForAppendLine extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows LineEditor to listen for a line of text and append that line to the line editor's buffer of
	 * strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		System.out.print("Type a line: ");
		Scanner theScannerForAppendLine = new Scanner(System.in);
		String theLine = theScannerForAppendLine.nextLine();
		//theScannerForAppendLine.close(); // closing this scanner breaks an input manager's scanner.
		
		try {
			LineEditor.bufferOfStrings.addLine(theLine);
		}
		catch (AnInsertsStringException theInsertsStringException) {
			System.out.println(theInsertsStringException.getMessage());
			return;
		}
		catch (AnInvalidCharacterException theInvalidCharacterException) {
			System.out.println(theInvalidCharacterException.getMessage());
			return;
		}
		
		System.out.print("The line editor appended \"" + theLine + "\" to its buffer of strings.\n\n");
		
	}
	
}