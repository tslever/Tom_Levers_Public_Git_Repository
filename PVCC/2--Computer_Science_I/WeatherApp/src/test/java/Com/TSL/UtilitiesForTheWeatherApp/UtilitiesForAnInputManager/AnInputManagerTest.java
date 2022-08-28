package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnInputManager;


import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;


/** ***********************************************************************************
 * AnInputManagerTest encapsulates methods to test the functionality of AnInputManager.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/25/21
 *********************************************************************************** */

public class AnInputManagerTest
{

    /** --------------------------------------------------------
     * testAnInputManager tests functionality of AnInputManager. 
     -------------------------------------------------------- */
    
    @Test
    public void testAnInputManager()
    {
        
        System.out.print ("Removing resources folder if it exists.\n\n");
        File theResourcesFolder = new File ("resources");
        if (theResourcesFolder.exists())
        {
            try
            {
                FileUtils.deleteDirectory (theResourcesFolder);
            }
            catch (IOException theIOException)
            {
                fail ("testAnInputManager failed to delete the resources folder.");
            }
        }
                
        try
        {
            System.out.println(
                "Creating input manager, database manager, resources folder, database, tables, and HTTP-request " +
                "manager."
            );
            AnInputManager theInputManager = new AnInputManager();
            System.out.println();
            
            System.out.println ("Executing command 'saved table'.");
            String[] theArrayOfComponentsOfTheCommand = new String[] {"saved", "table"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current table'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current", "table"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert zip'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert", "zip"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert zipco'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert", "zipco"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert 22903'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert", "22903"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);   
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert 19446'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert", "19446"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update zip'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update", "zip"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update zipco'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update", "zipco"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update 22222'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update", "22222"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update 22903'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update", "22903"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'delete'.");
            theArrayOfComponentsOfTheCommand = new String[] {"delete"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'delete zip'.");
            theArrayOfComponentsOfTheCommand = new String[] {"delete", "zip"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'delete zipco'.");
            theArrayOfComponentsOfTheCommand = new String[] {"delete", "zipco"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'delete 19446'.");
            theArrayOfComponentsOfTheCommand = new String[] {"delete", "19446"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'delete 22903'.");
            theArrayOfComponentsOfTheCommand = new String[] {"delete", "22903"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert 22903'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert", "22903"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update 22903'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update", "22903"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'now weather'.");
            theArrayOfComponentsOfTheCommand = new String[] {"now", "weather"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'now'.");
            theArrayOfComponentsOfTheCommand = new String[] {"now"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);

            System.out.println ("Executing command 'weekly weather'.");
            theArrayOfComponentsOfTheCommand = new String[] {"weekly", "weather"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'weekly'.");
            theArrayOfComponentsOfTheCommand = new String[] {"weekly"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);

            System.out.println ("Executing command 'probability precipitation'.");
            theArrayOfComponentsOfTheCommand = new String[] {"probability", "precipitation"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'probability'.");
            theArrayOfComponentsOfTheCommand = new String[] {"probability"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);

            System.out.println ("Executing command 'rate precipitation'.");
            theArrayOfComponentsOfTheCommand = new String[] {"rate", "precipitation"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'rate'.");
            theArrayOfComponentsOfTheCommand = new String[] {"rate"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'insert 19446'.");
            theArrayOfComponentsOfTheCommand = new String[] {"insert", "19446"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'update 19446'.");
            theArrayOfComponentsOfTheCommand = new String[] {"update", "19446"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'saved'.");
            theArrayOfComponentsOfTheCommand = new String[] {"saved"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'current'.");
            theArrayOfComponentsOfTheCommand = new String[] {"current"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'now'.");
            theArrayOfComponentsOfTheCommand = new String[] {"now"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'weekly'.");
            theArrayOfComponentsOfTheCommand = new String[] {"weekly"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'probability'.");
            theArrayOfComponentsOfTheCommand = new String[] {"probability"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'rate'.");
            theArrayOfComponentsOfTheCommand = new String[] {"rate"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command ''.");
            theArrayOfComponentsOfTheCommand = new String[] {""};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'unknown'.");
            theArrayOfComponentsOfTheCommand = new String[] {"unknown"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
            System.out.println ("Executing command 'exit app'.");
            theArrayOfComponentsOfTheCommand = new String[] {"exit", "app"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand); 
            
            System.out.println ("Executing command 'exit'.");
            theArrayOfComponentsOfTheCommand = new String[] {"exit"};
            theInputManager.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand); 
        }
        catch (SQLException theSqlException)
        {
            System.out.println ("SQL Exception: " + theSqlException.getMessage());
        }
        
    }
    
}
