package Com.TSL.LineEditorUtilities;


import java.util.Scanner;


/**
 * AnInputManager represents the structure for an input manager, which listens for and execute commands.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/27/21
 */

public class AnInputManager {

	
	/**
	 * listensForAndExecutesCommands listens for commands to the line editor and executes those commands.
	 * 
	 * @throws AnInvalidCommandException
	 * @throws ACommandMenuHasNotBeenSetUpException
	 */

	public void listensForAndExecutesCommands() throws AnInvalidCommandException, ACommandMenuHasNotBeenSetUpException {
		
		Scanner theScannerForACommand = new Scanner(System.in);
		
		while (true) {
		
			System.out.print("--> ");
			String theCommand = theScannerForACommand.nextLine();
			
			if (theCommand.equals("")) {
				System.out.print("The line editor received an empty command.\n\n");
				continue;
			}
			
			ACommandMenu theCommandMenu = TheCommandMenuGenerator.providesItsCommandMenu();
			
			String[] theArrayOfComponentsOfTheCommand = theCommand.split(" ", 3);
						
			switch (theArrayOfComponentsOfTheCommand[0].toLowerCase()) {
			
			case "a":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"a\" with too many arguments.\n\n");
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(
					new ACommand("Append line to the line editor's buffer of strings", "a", null)
				)
				.providesItsEncapsulatorForEdit()
				.edit(null);
				break;
				
				
			case "b":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"b\" with too many arguments.\n\n");
					continue;
				}
				
				System.out.print(LineEditor.bufferOfStrings + "\n\n");
				break;
				
				
			case "cls":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"cls\" with too many arguments.\n\n");
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Clear all lines from the line editor's buffer of strings", "cls", null
				))
				.providesItsEncapsulatorForEdit()
				.edit(null);
				break;
				
				
			case "del":
				if (theArrayOfComponentsOfTheCommand.length != 2) {
					System.out.print(
						"The line editor received command \"del\" without an index of a line to delete, or with too " +
						"many arguments.\n\n"
					);
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Delete the line in the buffer of strings with a given index", "del <index of line to delete>", null
				))
				.providesItsEncapsulatorForEdit()
				.edit(new String[] {theArrayOfComponentsOfTheCommand[1]});
				break;
				
				
			case "i":
				if (theArrayOfComponentsOfTheCommand.length != 2) {
					System.out.print(
						"The line editor received command \"i\" without an index at which to insert a string, or " +
						"with too many arguments.\n\n"
					);
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(
					new ACommand("Insert a given line at a given index", "i <index at which to insert>", null)
				)
				.providesItsEncapsulatorForEdit()
				.edit(new String[] {theArrayOfComponentsOfTheCommand[1]});
				break;
				
				
			case "line":
				if (theArrayOfComponentsOfTheCommand.length != 2) {
					System.out.print(
						"The line editor received command \"line\" without an index of a line to display, or with " +
						"too many arguments.\n\n"
					);
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Display the line in the buffer of strings with a given index",
					"line <index of line to display>",
					null
				))
				.providesItsEncapsulatorForEdit()
				.edit(new String[] {theArrayOfComponentsOfTheCommand[1]});
				break;
				

			case "lines":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"lines\" with too many arguments.\n\n");
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Display the number of lines of the line editor's buffer of strings", "lines", null
			    ))
				.providesItsEncapsulatorForEdit()
				.edit(null);
				break;
				
				
			case "load":
				if (theArrayOfComponentsOfTheCommand.length != 3) {
					System.out.print(
						"The line editor received command \"load\" without a path and an append / overwrite option.\n\n"
					);
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Load lines from a file into the line editor's buffer of strings",
					"load <path relative to project directory> <append / overwrite option (true / false)>",
					null
				))
				.providesItsEncapsulatorForEdit()
				.edit(new String[] {theArrayOfComponentsOfTheCommand[1], theArrayOfComponentsOfTheCommand[2]});
				break;
				
				
			case "m":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"m\" with too many arguments.\n\n");
					continue;
				}
				
				System.out.print(theCommandMenu + "\n\n");
				break;
				
				
			case "p":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"p\" with too many arguments.\n\n");
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(
					new ACommand("Print lines in the line editor's buffer of strings", "p", null)
				)
				.providesItsEncapsulatorForEdit()
				.edit(null);
				break;
				
				
			case "rep":
				if (theArrayOfComponentsOfTheCommand.length != 3) {
					System.out.print(
						"The line editor received command \"rep\" without a target string and a replacement string, " +
						"or with too many arguments.\n\n"
					);
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Replace all occurrences in each line in the line editor's buffer of strings of a first provided " +
					"string with a second provided string",
					"rep <target string> <replacement string>",
					null
				))
				.providesItsEncapsulatorForEdit()
				.edit(new String[] {theArrayOfComponentsOfTheCommand[1], theArrayOfComponentsOfTheCommand[2]});
				break;
				

			case "s":
				if (theArrayOfComponentsOfTheCommand.length != 2) {
					System.out.println(
						"The line editor received command \"save\" without a path, or with too many arguments."
					);
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand(
					"Save the lines in the line editor's buffer of strings to a file",
					"s <path relative to project directory>",
					null
				))
				.providesItsEncapsulatorForEdit()
				.edit(new String[] {theArrayOfComponentsOfTheCommand[1]});
				break;
				
				
			case "quit":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"q\" with too many arguments.\n\n");
					continue;
				}
				
				theScannerForACommand.close();
				theCommandMenu
				.providesTheFirstInstanceOf(new ACommand("Quit the line editor", "quit", null))
				.providesItsEncapsulatorForEdit()
				.edit(null);
				break;
				
				
			case "words":
				if (theArrayOfComponentsOfTheCommand.length != 1) {
					System.out.print("The line editor received the command \"words\" with too many arguments.\n\n");
					continue;
				}
				
				theCommandMenu
				.providesTheFirstInstanceOf(
					new ACommand("Display the number of words in the line editor's buffer of strings", "words", null)
				)
				.providesItsEncapsulatorForEdit()
				.edit(null);
				break;
				
			}
		
		}
		
	}

}
