package Com.TSL.LineEditorUtilities;


import java.io.IOException;


/**
 * LineEditor encapsulates a buffer of strings and the entry point of this program, which offers file-manipulation and
 * line-editing functionality.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/27/21
 */

public class LineEditor 
{
	
	public static final LineList bufferOfStrings = new LineList();
	
	
	/**
	 * main is the entry point of this program, which offers file-manipulation and line-editing functionality.
	 * 
	 * @param args
	 * @throws AnInsertsStringException
	 * @throws ACommandMenuHasBeenSetUpException
	 * @throws ACommandMenuHasNotBeenSetUpException
	 * @throws AnInsertsCommandException
	 * @throws AnInvalidCharacterException
	 * @throws AnInvalidCommandException
	 * @throws IOException
	 */
	
    public static void main(String[] args) throws
    	AnInsertsStringException,
    	ACommandMenuHasBeenSetUpException,
    	ACommandMenuHasNotBeenSetUpException,
    	AnInsertsCommandException,
    	AnInvalidCharacterException,
    	AnInvalidCommandException,
    	IOException
    {
    	
    	System.out.print("Welcome to Line Editor.\n\n");
    
    	if (args.length != 0) {
    		bufferOfStrings.loadsTheFileAt(args[0]);
    	}
    	
    	TheCommandMenuGenerator.setsUpTheCommandMenu();
    	System.out.print(TheCommandMenuGenerator.providesItsCommandMenu() + "\n\n");
    	
    	(new AnInputManager()).listensForAndExecutesCommands();
    	
    }
    
}