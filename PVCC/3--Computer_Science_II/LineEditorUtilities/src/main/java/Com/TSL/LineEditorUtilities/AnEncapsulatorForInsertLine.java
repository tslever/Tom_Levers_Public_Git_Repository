package Com.TSL.LineEditorUtilities;

import java.util.Scanner;

/**
 * AnEncapsulatorForInsertLine encapsulates a version of the edit method by which the line editor inserts a line at a
 * given index in the line editor's buffer of strings.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/26/21
 */

public class AnEncapsulatorForInsertLine extends AnEncapsulatorForEdit {
	
	
	/**
	 * edit allows the line editor to listen for a line of text and append that line to the line editor's buffer of
	 * strings.
	 */
	
	public void edit(String[] theArrayOfArguments) {
		
		System.out.print("Type a line: ");
		Scanner theScannerForInsertLine = new Scanner(System.in);
		String theLine = theScannerForInsertLine.nextLine();
		//theScannerForAppendLine.close(); // closing this scanner breaks an input manager's scanner.
		
		try {
			LineEditor.bufferOfStrings.addLine(theLine, Integer.parseInt(theArrayOfArguments[0]));
		}
		catch (AnInsertsStringException theInsertsStringException) {
			System.out.println(theInsertsStringException.getMessage());
			return;
		}
		catch (AnInvalidCharacterException theInvalidCharacterException) {
			System.out.println(theInvalidCharacterException.getMessage());
			return;
		}
		catch (NumberFormatException theNumberFormatException) {
			System.out.println(theNumberFormatException.getMessage());
			return;
		}
		
	}
	
}