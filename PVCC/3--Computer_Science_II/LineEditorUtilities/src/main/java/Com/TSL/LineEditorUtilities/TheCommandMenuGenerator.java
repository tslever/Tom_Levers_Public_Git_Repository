package Com.TSL.LineEditorUtilities;


/**
 * TheCommandMenuGenerator encapsulates a command menu, an indicator of whether or not the command menu has been set up,
 * a method to set up the command menu, and a method to provide the command menu.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/27/21
 */

public class TheCommandMenuGenerator {

	
	private static ACommandMenu commandMenu;
	private static boolean theCommandMenuHasBeenSetUp = false;
	
	
	/**
	 * setsUpTheCommandMenu sets up this generator's command menu, if the command menu has not been set up already.
	 * 
	 * @throws ACommandMenuHasBeenSetUpException
	 * @throws AnInsertsCommandException
	 */
	
	public static void setsUpTheCommandMenu() throws ACommandMenuHasBeenSetUpException, AnInsertsCommandException {
		
		if (theCommandMenuHasBeenSetUp) {
			throw new ACommandMenuHasBeenSetUpException(
				"The command-menu generator tried to set up a command menu that was already set up."
			);
		}
		
    	commandMenu = new ACommandMenu();
    	
    	commandMenu.inserts(
    		new ACommand("Append line to the line editor's buffer of strings", "a", new AnEncapsulatorForAppendLine())
    	);
    	
    	commandMenu.inserts(new ACommand(
    		"Clear all lines from the line editor's buffer of strings", "cls", new AnEncapsulatorForClearBuffer()
    	));
    	
    	commandMenu.inserts(new ACommand(
    		"Display the number of words in the line editor's buffer of strings",
    		"words",
    		new AnEncapsulatorForCountWords()
    	));
    	
    	commandMenu.inserts(new ACommand(
    		"Delete the line in the buffer of strings with a given index",
    		"del <index of line to delete>",
    		new AnEncapsulatorForDeleteLine())
    	);
    	
    	commandMenu.inserts(new ACommand(
    		"Display the line in the buffer of strings with a given index",
    		"line <index of line to display>",
    		new AnEncapsulatorForDisplayLine()
    	));
    	
    	commandMenu.inserts(new ACommand(
    		"Display the number of lines of the line editor's buffer of strings",
    		"lines",
    		new AnEncapsulatorForGetNumberOfLines()
    	));
    	
    	commandMenu.inserts(new ACommand(
    		"Insert a given line at a given index", "i <index at which to insert>", new AnEncapsulatorForInsertLine()
    	));
    	
    	commandMenu.inserts(new ACommand(
			"Load lines from a file into the line editor's buffer of strings",
			"load <path relative to project directory> <append / overwrite option (true / false)>",
			new AnEncapsulatorForLoadFile()
		));
			
    	commandMenu.inserts(new ACommand("Display this command menu", "m", null));
    	
    	commandMenu.inserts(
    		new ACommand("Print lines in the line editor's buffer of strings", "p", new AnEncapsulatorForPrintLines())
    	);
    	
    	commandMenu.inserts(new ACommand("Quit the line editor", "quit", new AnEncapsulatorForQuit()));
    	
    	commandMenu.inserts(new ACommand(
			"Replace all occurrences in each line in the line editor's buffer of strings of a first provided string " +
			"with a second provided string",
			"rep <target string> <replacement string>",
			new AnEncapsulatorForReplaceStrings()
		));

    	commandMenu.inserts(new ACommand(
    		"Save the lines in the line editor's buffer of strings to a file",
    		"s <path relative to project directory>",
    		new AnEncapsulatorForSaveLines())
    	);
    	
    	theCommandMenuHasBeenSetUp = true;
		
	}
	
	
	/**
	 * providesItsCommandMenu provides the command menu of this generator.
	 * 
	 * @return
	 * @throws ACommandMenuHasNotBeenSetUpException
	 */
	
	public static ACommandMenu providesItsCommandMenu() throws ACommandMenuHasNotBeenSetUpException {
		
		if (!theCommandMenuHasBeenSetUp) {
			throw new ACommandMenuHasNotBeenSetUpException(
				"The command-menu generator tried to provide a command menu that was not set up."
			);
		}
    	
    	return commandMenu;
		
	}
	
}
