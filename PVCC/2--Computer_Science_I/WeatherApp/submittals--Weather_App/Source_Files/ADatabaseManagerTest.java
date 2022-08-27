package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForADatabaseManager;


import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import org.apache.commons.io.FileUtils;
import org.junit.Assert;
import org.junit.jupiter.api.Test;


/** ***************************************************************************************
 * ADatabaseManagerTest encapsulates methods to test the functionality of ADatabaseManager.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/25/21
 ************************************************************************************** */

public class ADatabaseManagerTest {

    
    /** ------------------------------------------------------------
     * testADatabaseManager tests functionality of ADatabaseManager. 
     ------------------------------------------------------------ */
    
    @Test
    public void testADatabaseManager()
    {
        
        System.out.print("Removing resources folder if it exists.\n\n");
        File theResourcesFolder = new File ("resources");
        if (theResourcesFolder.exists())
        {
            try
            {
                FileUtils.deleteDirectory (theResourcesFolder);
            }
            catch (IOException theIOException)
            {
                Assert.fail ("testADatabaseManager failed to delete the resources folder.");
            }
        }
        
        try
        {
            
            System.out.print ("Creating database manager, resources folder, database, and tables.\n\n");
            ADatabaseManager theDatabaseManager = new ADatabaseManager();
            
            System.out.print(
                "Displaying the table for the current zip code and the table for the saved zip codes.\n\n"
            );
            theDatabaseManager.displaysTheTableForTheCurrentZipCode();
            theDatabaseManager.displaysTheTableForTheSavedZipCodes();
            
            System.out.println ("Checking valid and invalid zip codes.");
            try
            {
                theDatabaseManager.checks ("0");
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                System.out.println ("An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage());
            }
            
            try
            {
                theDatabaseManager.checks ("2290.");
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                System.out.println ("An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage());
            }
            
            try
            {
                String theZipCode = "22903";
                theDatabaseManager.checks (theZipCode);
                System.out.println (theZipCode + " is a valid zip code.");
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                System.out.println ("An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage());
            }
            System.out.println();
            
            System.out.println ("Trying to delete a zip code from the table for the saved zip codes.");
            try
            {
                theDatabaseManager.deletesFromTheTableOfSavedZipCodes ("22903");
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                System.out.print(
                    "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                );
            }
            
            System.out.println ("Inserting a zip code into the table for the saved zip codes.");
            try
            {
                theDatabaseManager.insertsIntoTheTableOfSavedZipCodes ("22903");
                theDatabaseManager.displaysTheTableForTheSavedZipCodes();
                theDatabaseManager.displaysTheTableForTheCurrentZipCode();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                Assert.fail(
                    "A database manager failed to insert a zip code into the table for the saved zip codes."
                );
            }
            
            System.out.println(
                "Trying to insert a zip code already in the table for the saved zip codes into that table."
            );
            try
            {
                theDatabaseManager.insertsIntoTheTableOfSavedZipCodes ("22903");
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                System.out.print(
                    "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                );
            }
            
            System.out.println ("Deleting a zip code from the table for the saved zip codes.");
            try
            {
                theDatabaseManager.deletesFromTheTableOfSavedZipCodes ("22903");
                theDatabaseManager.displaysTheTableForTheSavedZipCodes();
                theDatabaseManager.displaysTheTableForTheCurrentZipCode();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                Assert.fail ("A database manager failed to delete a zip code from the table for the saved zip codes");
            }
            
            System.out.println ("Trying to get the current zip code when the current zip code has not been set.");
            try
            {
                theDatabaseManager.providesTheCurrentZipCode();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
               System.out.println ("An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage());
               System.out.println();
            }
            
            
            System.out.println(
                "Trying to update the current zip code with a zip code that is not in the table of saved zip codes."
            );
            try
            {
                theDatabaseManager.updatesTheTableForTheCurrentZipCodeWith ("22903");
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                System.out.println ("An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage());
                System.out.println();
            }
            
            System.out.println ("Inserting a zip code into the table for the saved zip codes.");
            try
            {
                theDatabaseManager.insertsIntoTheTableOfSavedZipCodes ("22903");
                theDatabaseManager.displaysTheTableForTheSavedZipCodes();
                theDatabaseManager.displaysTheTableForTheCurrentZipCode();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                Assert.fail(
                    "A database manager failed to insert a zip code into the table for the saved zip codes."
                );
            }
            
            System.out.println(
                "Updating the current zip code with a zip code that is in the table of saved zip codes."
            );
            try
            {
                theDatabaseManager.updatesTheTableForTheCurrentZipCodeWith ("22903");
                theDatabaseManager.displaysTheTableForTheSavedZipCodes();
                theDatabaseManager.displaysTheTableForTheCurrentZipCode();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                Assert.fail ("A database manager failed to update the current zip code.");
            }
            
            System.out.println ("Getting the current zip code when the current zip code has been set.");
            try
            {
                System.out.println (theDatabaseManager.providesTheCurrentZipCode());
                System.out.println();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
               Assert.fail ("A database manager failed to get the current zip code.");
            }
            
            
            System.out.println(
                "Deleting a zip code from the table for the saved zip codes, and removing the zip code the table " +
                "for the current zip code."
            );
            try
            {
                theDatabaseManager.deletesFromTheTableOfSavedZipCodes ("22903");
                theDatabaseManager.displaysTheTableForTheSavedZipCodes();
                theDatabaseManager.displaysTheTableForTheCurrentZipCode();
            }
            catch (AnInvalidZipCodeException theInvalidZipCodeException)
            {
                Assert.fail(
                    "A database manager failed to delete a zip code from the table for the saved zip codes."
                );
            }
            
        }
        catch (SQLException theSQLException)
        {
            Assert.fail (theSQLException.getMessage());
        }
        
    }
    
}
