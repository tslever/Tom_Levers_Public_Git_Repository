package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForADatabaseManager;


import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;


/** ****************************************************************************************************************
 * ADatabaseManager represents the structure for a database manager, which allows manipulating current and saved zip
 * codes.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/28/21
 **************************************************************************************************************** */

public class ADatabaseManager
{

    private final String PATH_TO_THE_FOLDER_FOR_THE_DATABASE = "resources";
    private final String PATH_TO_THE_DATABASE =
        this.PATH_TO_THE_FOLDER_FOR_THE_DATABASE + "/The_Database_For_The_Current_Zip_Code_And_Saved_Zip_Codes.db";
    
    
    /** -------------------------------------------------------------------------------------------------------
     * ADatabaseManager() is the zero-parameter constructor for ADatabaseManager, which conditionally creates a
     * database.
     * 
     * @throws SQLException
     ------------------------------------------------------------------------------------------------------- */
    
    public ADatabaseManager() throws SQLException
    {
        this.createsTheDatabase();
    }
    
    
    /** ---------------------------------------------------------------------------------------------------------
     * checks throws an invalid zip code exception if a provided zip code does not have the appropriate number of
     * digits, or contains characters that are not digits.
     * 
     * @param theZipCode
     * @throws AnInvalidZipCodeException
     --------------------------------------------------------------------------------------------------------- */
    
    // TODO: Make private after testing.
    protected void checks (String theZipCode) throws AnInvalidZipCodeException
    {
        if (theZipCode.length() != 5)
        {
            throw new AnInvalidZipCodeException ("The number of characters in a zip code is not equal to 5.");
        }
        
        Character theCharacter;
        for (int i = 0; i < theZipCode.length(); i++)
        {
            theCharacter = theZipCode.charAt (i);
            if (!Character.isDigit (theCharacter))
            {
                throw new AnInvalidZipCodeException ("A character in a zip code is not a digit.");
            }
        }
    }
    
    
    /** -----------------------------------------------------------------------------------------------------------
     * createsTheDatabase creates a folder at a specified path for a database if the folder does not already exist,
     * and creates the database if it doesn't already exist. 
     * 
     * @throws SQLException
     ----------------------------------------------------------------------------------------------------------- */
    
    private void createsTheDatabase() throws SQLException
    {
        File theFolder = new File (this.PATH_TO_THE_FOLDER_FOR_THE_DATABASE);
        if (!theFolder.exists())
        {
            theFolder.mkdir();
        }
        
        File theFile = new File (this.PATH_TO_THE_DATABASE);
        if (!theFile.exists())
        {
            AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
            theConnectionAndTheStatement.close();
        }
        
        this.createsTheTablesInTheDatabase();
    }
    
    
    /** ---------------------------------------------------------------------------------------------------------------
     * createsTheTablesInTheDatabase creates a table for a current zip code, if it doesn't already exist, inserts a row
     * with key 0 and a null zip code if a row with key 0 does not already exist, and creates a table for saved zip
     * codes, if it doesn't already exist.
     * 
     * @throws SQLException
     --------------------------------------------------------------------------------------------------------------- */
    
    private void createsTheTablesInTheDatabase() throws SQLException
    {
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery =
            "CREATE TABLE IF NOT EXISTS The_Table_For_The_Current_Zip_Code (\n" +
            "the_id_of_the_current_zip_code INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
            "the_value_of_the_current_zip_code VARCHAR(5) NOT NULL UNIQUE\n" +
            ");";
        theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        
        theQuery =
            "INSERT OR IGNORE INTO The_Table_For_The_Current_Zip_Code\n" +
            "(the_id_of_the_current_zip_code, the_value_of_the_current_zip_code)\n" +
            "VALUES (0, '-----');";
        theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        
        theQuery =
            "CREATE TABLE IF NOT EXISTS The_Table_For_The_Saved_Zip_Codes (\n" +
            "the_saved_zip_codes VARCHAR(5) NOT NULL PRIMARY KEY UNIQUE\n" +
            ");";
        theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        
        theConnectionAndTheStatement.close();
        
    }
    
    
    /** ----------------------------------------------------------------------------------------------------------------
     * deletesFromTheTableOfSavedZipCodes deletes a provided zip code from the table of saved zip codes if the provided
     * zip code exists in the table of saved zip codes, or throws an invalid zip code exception if the provided zip code
     * is invalid or does not exist in the table of saved zip codes.
     * 
     * @param theZipCode
     * @throws AnInvalidZipCodeException
     * @throws SQLException
     ---------------------------------------------------------------------------------------------------------------- */
    
    public void deletesFromTheTableOfSavedZipCodes (String theZipCode) throws AnInvalidZipCodeException, SQLException
    {
        this.checks (theZipCode);
        
        if (!theTableForTheSavedZipCodesContains (theZipCode))
        {
            throw new AnInvalidZipCodeException(
                "A database manager received a request to remove a zip code from the table of saved zip codes that " +
                "wasn't in the table of saved zip codes."
            );
        }
        
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery;
        if (theTableForTheCurrentZipCodeContains (theZipCode))
        {
            theQuery =
                "UPDATE The_Table_For_The_Current_Zip_Code\n" +
                "SET the_value_of_the_current_zip_code = '-----'\n" +
                "WHERE the_id_of_the_current_zip_code = 0;";
            theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        }
        
        theQuery =
            "DELETE FROM The_Table_For_The_Saved_Zip_Codes WHERE the_saved_zip_codes = " + theZipCode + ";";
        theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        
        theConnectionAndTheStatement.close();
        
    }
    
    
    /** --------------------------------------------------------------------------------------------------
     * displaysTheTableForTheCurrentZipCode displays the components of the table for the current zip code.
     * 
     * @throws SQLException
     -------------------------------------------------------------------------------------------------- */
    
    public void displaysTheTableForTheCurrentZipCode() throws SQLException
    {
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery = "SELECT * FROM The_Table_For_The_Current_Zip_Code";
        ResultSet theResultSet = theConnectionAndTheStatement.providesItsStatement().executeQuery (theQuery);
        
        System.out.println(
            "The_Table_For_The_Current_Zip_Code\nthe_id_of_the_current_zip_code | the_value_of_the_current_zip_code"
        );
        while (theResultSet.next())
        {
            System.out.println(
                theResultSet.getInt("the_id_of_the_current_zip_code") + " | " +
                theResultSet.getString("the_value_of_the_current_zip_code")
            );
        }
        System.out.println();
        
        theConnectionAndTheStatement.close();
    }
    
    
    /** ------------------------------------------------------------------------------------------------
     * displaysTheTableForTheSavedZipCodes displays the components of the table for the saved zip codes.
     * 
     * @throws SQLException
     ------------------------------------------------------------------------------------------------ */
    
    public void displaysTheTableForTheSavedZipCodes() throws SQLException
    {
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery = "SELECT * FROM The_Table_For_The_Saved_Zip_Codes";
        ResultSet theResultSet = theConnectionAndTheStatement.providesItsStatement().executeQuery (theQuery);
        
        System.out.println("The Table for the Saved Zip Codes\nthe saved zip codes");
        while (theResultSet.next())
        {
            System.out.println(theResultSet.getString ("the_saved_zip_codes"));
        }
        System.out.println();
        
        theConnectionAndTheStatement.close();
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------
     * insertsIntoTheTableOfSavedZipCodes(String theZipCode) inserts a provided zip code into the table of saved zip
     * codes if the zip code is valid and does not exist already in the table of saved zip codes, or throws an invalid
     * zip code exception if the zip code is invalid or is in the table of saved zip codes already.
     * 
     * @param theZipCode
     * @throws AnInvalidZipCodeException
     * @throws SQLException
     ------------------------------------------------------------------------------------------------------------- */
    
    public void insertsIntoTheTableOfSavedZipCodes (String theZipCode) throws AnInvalidZipCodeException, SQLException
    {
        this.checks (theZipCode);
        
        if (theTableForTheSavedZipCodesContains (theZipCode))
        {
            throw new AnInvalidZipCodeException(
                "A database manager received a request to insert a zip code into the table of saved zip codes that " +
                "was already in the table of saved zip codes."
            );
        }
        
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery =
            "INSERT INTO The_Table_For_The_Saved_Zip_Codes\n" +
            "(the_saved_zip_codes)\n" +
            "VALUES (" + theZipCode + ");";
        theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        
        theConnectionAndTheStatement.close();
        
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------
     * providesTheCurrentZipCode provides the current zip code, or throws an invalid zip code exception if the current
     * zip code does not exist.
     * 
     * @return
     * @throws AnInvalidZipCodeException
     * @throws SQLException
     -------------------------------------------------------------------------------------------------------------- */
    
    public String providesTheCurrentZipCode() throws AnInvalidZipCodeException, SQLException
    {
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery = "SELECT * FROM The_Table_For_The_Current_Zip_Code";
        ResultSet theResultSet = theConnectionAndTheStatement.providesItsStatement().executeQuery (theQuery);
        
        String theCurrentZipCode = theResultSet.getString ("the_value_of_the_current_zip_code");
        theConnectionAndTheStatement.close();        
        
        if (theCurrentZipCode.equals ("-----"))
        {
            throw new AnInvalidZipCodeException ("The retrieved current zip code was '-----'.");
        }
        
        return theCurrentZipCode;
    }
    
    
    /** ----------------------------------------------------------------------------------------------------------
     * theTableForTheCurrentZipCodeContains indicates whether or not the table for the current zip code contains a
     * provided zip code, or throws an invalid zip code exception if the provided zip code is invalid.
     * 
     * @param theZipCode
     * @return
     * @throws AnInvalidZipCodeException
     * @throws SQLException
     ---------------------------------------------------------------------------------------------------------- */
    
    private boolean theTableForTheCurrentZipCodeContains (String theZipCode)
        throws AnInvalidZipCodeException, SQLException
    {
        this.checks (theZipCode);
        
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery =
            "SELECT COUNT(*) AS the_number_of_matching_zip_codes\n" +
            "FROM The_Table_For_The_Current_Zip_Code\n" +
            "WHERE the_value_of_the_current_zip_code = " + theZipCode + ";";
        ResultSet theResultSet = theConnectionAndTheStatement.providesItsStatement().executeQuery (theQuery);
        
        int theNumberOfMatchingZipCodes = theResultSet.getInt ("the_number_of_matching_zip_codes");
        theConnectionAndTheStatement.close();
        
        if (theNumberOfMatchingZipCodes > 0)
        {
            return true;
        }
        else
        {
            return false;
        }
        
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------
     * theTableForTheSavedZipCodesContains(String theZipCode) indicates whether or not the table for the current zip
     * code contains a provided zip code, or throws an invalid zip code exception if the provided zip code is invalid.
     * 
     * @param theZipCode
     * @return
     * @throws AnInvalidZipCodeException
     * @throws SQLException
     ------------------------------------------------------------------------------------------------------------- */
    
    private boolean theTableForTheSavedZipCodesContains (String theZipCode)
        throws AnInvalidZipCodeException, SQLException
    {
        this.checks (theZipCode);
        
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        String theQuery =
            "SELECT COUNT(*) AS the_number_of_matching_zip_codes\n" +
            "FROM The_Table_For_The_Saved_Zip_Codes\n" +
            "WHERE the_saved_zip_codes = " + theZipCode + ";"; 
        ResultSet theResultSet = theConnectionAndTheStatement.providesItsStatement().executeQuery (theQuery);
        
        int theNumberOfMatchingZipCodes = theResultSet.getInt ("the_number_of_matching_zip_codes");
        theConnectionAndTheStatement.close();
        
        if (theNumberOfMatchingZipCodes > 0)
        {
            return true;
        }
        else
        {
            return false;
        }
        
    }
    
    
    /** -----------------------------------------------------------------------------------------------------
     * providesTheConnectionAndTheStatement provides a connection and a statement in an encapsulating object.
     * 
     * @return
     * @throws SQLException
     ----------------------------------------------------------------------------------------------------- */
    
    private AConnectionAndAStatement providesTheConnectionAndTheStatement() throws SQLException
    {
        Connection theConnection = DriverManager.getConnection ("jdbc:sqlite:" + this.PATH_TO_THE_DATABASE);
        Statement theStatement = theConnection.createStatement();
        
        return new AConnectionAndAStatement (theConnection, theStatement);
    }
    
    
    /** ---------------------------------------------------------------------------------------------------------------
     * updatesTheTableForTheCurrentZipCodeWith(String theZipCode) updates the current zip code with a provided zip code
     * if the provided zip code exists in the table of saved zip codes, or throws an invalid zip code exception if the
     * provided zip code is invalid or is not in the table for the saved zip codes.
     * 
     * @param theZipCode
     * @throws AnInvalidZipCodeException
     * @throws SQLException
     --------------------------------------------------------------------------------------------------------------- */
    
    public void updatesTheTableForTheCurrentZipCodeWith (String theZipCode)
        throws AnInvalidZipCodeException, SQLException
    {
        this.checks (theZipCode);
        
        AConnectionAndAStatement theConnectionAndTheStatement = this.providesTheConnectionAndTheStatement();
        
        if (!theTableForTheSavedZipCodesContains (theZipCode))
        {
            throw new AnInvalidZipCodeException(
                "The table of saved zip codes does not contain the potential current zip code."
            );
        }
        
        String theQuery =
            "UPDATE The_Table_For_The_Current_Zip_Code\n" +
            "SET the_value_of_the_current_zip_code = " + theZipCode +
            "\nWHERE the_id_of_the_current_zip_code = 0;";
        theConnectionAndTheStatement.providesItsStatement().execute (theQuery);
        
        theConnectionAndTheStatement.close();
    }
    
}
