package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnInputManager;


import Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForADatabaseManager.ADatabaseManager;
import Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForADatabaseManager.AnInvalidZipCodeException;
import Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAParser.AParserForAJsonObjectRepresentingForecasts;
import Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnHttpRequestManager.AnHttpRequestManager;
import java.io.IOException;
import java.net.MalformedURLException;
import java.sql.SQLException;
import java.util.Scanner;


/** *********************************************************************************************************************
 * AnInputManager represents the structure for an input manager, which listens for commands from a user, parses them, and
 * completes appropriate functionality.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/28/21
 ********************************************************************************************************************* */

public class AnInputManager
{

    private ADatabaseManager databaseManager;
    private AnHttpRequestManager httpRequestManager;
    
    
    /** -------------------------------------------------------------------------------------------------------------------
     * AnInputManager() is the zero-parameter constructor for AnInputManager, which sets the database manager of this input
     * manager to a new database manager, and sets the HTTP-request manager of this input manager to a new HTTP-request
     * manager.
     * 
     * @throws SQLException
     ------------------------------------------------------------------------------------------------------------------- */
    
    public AnInputManager() throws SQLException
    {
        this.databaseManager = new ADatabaseManager();
        this.httpRequestManager = new AnHttpRequestManager();
    }
    
    
    /** -------------------------------------------------------------------------
     * displaysAnIntroduction displays an introduction to use of the Weather App.
     ------------------------------------------------------------------------- */
    
    public void displaysAnIntroduction()
    {
        System.out.print ("Welcome to the Weather App.\n\n");
        
        this.displaysTheCommandMenu();
    }
    
    
    /** -------------------------------------------------------------------
     * displaysTheCommandMenu displays the command menu of the Weather App.
     ------------------------------------------------------------------- */
    
    private void displaysTheCommandMenu()
    {
        System.out.print(
            "Command Menu\n" +
            "current: Display the current zip code.\n" +
            "delete <a zip code>: Remove a tracked location from the table of saved zip codes.\n" +
            "exit: Exit the weather app.\n" +
            "insert <a zip code>: Insert a zip code into the table of saved zip codes.\n" +
            "menu: Display this command menu.\n" +
            "now: Display a detailed weather forecast for the current day at the current location.\n" +
            "probability: Display the percentage chance that it is raining right now at the current location.\n" +
            "rate: Display average volume precipitation per unit area for this minute.\n" +
            "saved: Display the table of saved zip codes.\n" +
            "update <a zip code>: Update the current zip code with a zip code.\n" +
            "weekly: Display a weather forecast for the current week at the current location.\n\n"
        );
    }
    
    
    /** -------------------------------------------------------------------------------------------------------
     * executesTheCommandRepresentedBy executes the command represented by an array of components of a command.
     * 
     * @param theArrayOfComponentsOfTheCommand
     ------------------------------------------------------------------------------------------------------- */
    
    // TODO: Make private after testing.
    protected void executesTheCommandRepresentedBy (String[] theArrayOfComponentsOfTheCommand)
    {
        switch (theArrayOfComponentsOfTheCommand[0].toLowerCase())
        {
        
        case "current":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"current\" had an argument.\n\n");
            }
            else
            {
                try
                {
                    System.out.print(
                        "The current zip code is " + this.databaseManager.providesTheCurrentZipCode() + ".\n\n"
                    );
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (SQLException theSqlException)
                {
                    System.out.print("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;                
        
        case "delete":
            if (theArrayOfComponentsOfTheCommand.length == 1)
            {
                System.out.print("The delete command needs a zip code.\n\n");
            }
            else
            {
                try
                {
                    this.databaseManager.deletesFromTheTableOfSavedZipCodes (theArrayOfComponentsOfTheCommand[1]);
                    System.out.print (theArrayOfComponentsOfTheCommand[1] + " was deleted.\n\n");
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (SQLException theSqlException)
                {
                    System.out.print("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "exit":
            if (!theExitCommandIsRepresentedBy (theArrayOfComponentsOfTheCommand))
            {
                System.out.print ("A command with action \"exit\" had an argument.\n\n");
            }
            else
            {
                System.out.print ("Exiting the weather app.\n\n");
                System.exit (AnExitStatus.SUCCESS.ordinal());
            }
            break;
        
        case "insert":
            if (theArrayOfComponentsOfTheCommand.length == 1)
            {
                System.out.print ("The insert command needs a zip code.\n\n");
            }
            else
            {
                try
                {
                    this.databaseManager.insertsIntoTheTableOfSavedZipCodes (theArrayOfComponentsOfTheCommand[1]);
                    System.out.print(
                        theArrayOfComponentsOfTheCommand[1] + " was inserted into the table of saved zip codes.\n\n"
                    );
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (SQLException theSqlException)
                {
                    System.out.println("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "menu":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"menu\" had an argument.\n\n");
            }
            else
            {
                this.displaysTheCommandMenu();
            }
            break;
            
        case "now":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"now\" had an argument.\n\n");
            }
            else
            {
                try
                {
                    AParserForAJsonObjectRepresentingForecasts theParserForAJsonObjectRepresentingForecasts =
                        new AParserForAJsonObjectRepresentingForecasts(
                            this.httpRequestManager.providesAJsonObjectRepresentingForecastsBasedOn(
                                this.databaseManager.providesTheCurrentZipCode()
                            )
                        );
                    
                    theParserForAJsonObjectRepresentingForecasts.displaysTheCurrentWeather();
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (MalformedURLException theMalformedUrlException)
                {
                    System.out.print(
                        "Malformed URL Exception: " + theMalformedUrlException.getMessage() + "\n\n"
                    );
                }
                catch (IOException theIoException) // A superclass of MalformedURLException.
                {
                    System.out.println ("IO Exception: " + theIoException.getMessage() + "\n\n");
                }
                catch (SQLException theSqlException)
                {
                    System.out.print ("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "probability":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"probability\" had an argument.\n\n");
            }
            else
            {
                try
                {
                    AParserForAJsonObjectRepresentingForecasts theParserForAJsonObjectRepresentingForecasts =
                    new AParserForAJsonObjectRepresentingForecasts(
                        this.httpRequestManager.providesAJsonObjectRepresentingForecastsBasedOn(
                            this.databaseManager.providesTheCurrentZipCode()
                        )
                    );
                
                    theParserForAJsonObjectRepresentingForecasts.displaysTheProbabilityOfPrecipitationToday();
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (MalformedURLException theMalformedUrlException)
                {
                    System.out.print(
                        "Malformed URL Exception: " + theMalformedUrlException.getMessage() + "\n\n"
                    );
                }
                catch (IOException theIoException) // A superclass of MalformedURLException.
                {
                    System.out.print("IO Exception: " + theIoException.getMessage() + "\n\n");
                }
                catch (SQLException theSqlException)
                {
                    System.out.print("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "rate":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"rate\" had an argument.\n\n");
            }
            else
            {
                try
                {
                    AParserForAJsonObjectRepresentingForecasts theParserForAJsonObjectRepresentingForecasts =
                    new AParserForAJsonObjectRepresentingForecasts(
                        this.httpRequestManager.providesAJsonObjectRepresentingForecastsBasedOn(
                            this.databaseManager.providesTheCurrentZipCode()
                        )
                    );
                
                    theParserForAJsonObjectRepresentingForecasts
                        .displaysTheAverageVolumePrecipitationPerUnitAreaForThisMinute();
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (MalformedURLException theMalformedUrlException)
                {
                    System.out.print(
                        "Malformed URL Exception: " + theMalformedUrlException.getMessage() + "\n\n"
                    );
                }
                catch (IOException theIoException)
                {
                    System.out.print("IO Exception: " + theIoException.getMessage() + "\n\n");
                }
                catch (SQLException theSqlException)
                {
                    System.out.print("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "saved":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"saved\" had an argument.\n\n");
            }
            else
            {
                try
                {
                    this.databaseManager.displaysTheTableForTheSavedZipCodes();
                }
                catch (SQLException theSqlException)
                {
                    System.out.print ("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "update":
            if (theArrayOfComponentsOfTheCommand.length == 1)
            {
                System.out.print ("The update command needs a zip code.\n\n");
            }
            else
            {
                try
                {
                    this.databaseManager.updatesTheTableForTheCurrentZipCodeWith(theArrayOfComponentsOfTheCommand[1]);
                    System.out.print(
                        "The current zip code was updated to be " + theArrayOfComponentsOfTheCommand[1] + ".\n\n"
                    );
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (SQLException theSqlException)
                {
                    System.out.print("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        case "weekly":
            if (theArrayOfComponentsOfTheCommand.length != 1)
            {
                System.out.print ("A command with action \"weekly\" had an argument.\n\n");
            }
            else
            {
                try
                {
                    AParserForAJsonObjectRepresentingForecasts theParserForAJsonObjectRepresentingForecasts =
                    new AParserForAJsonObjectRepresentingForecasts(
                        this.httpRequestManager.providesAJsonObjectRepresentingForecastsBasedOn(
                            this.databaseManager.providesTheCurrentZipCode()
                        )
                    );
                
                    theParserForAJsonObjectRepresentingForecasts.displaysTheWeeklyForecast();
                }
                catch (AnInvalidZipCodeException theInvalidZipCodeException)
                {
                    System.out.print(
                        "An Invalid Zip Code Exception: " + theInvalidZipCodeException.getMessage() + "\n\n"
                    );
                }
                catch (MalformedURLException theMalformedUrlException)
                {
                    System.out.print(
                        "Malformed URL Exception: " + theMalformedUrlException.getMessage() + "\n\n"
                    );
                }
                catch (IOException theIoException)
                {
                    System.out.println("IO Exception: " + theIoException.getMessage() + "\n\n");
                }
                catch (SQLException theSqlException)
                {
                    System.out.print("SQL Exception: " + theSqlException.getMessage() + "\n\n");
                }
            }
            break;
            
        default:
            System.out.print("Try another command.\n\n");
            
        }
    }
    
    
    /** -----------------------------------------------------------------------------------
     * listensForCommands listens for commands from a user, parses them, and executes them.
     ----------------------------------------------------------------------------------- */
    
    public void listensForCommands()
    {
        
        Scanner theScannerForACommand = new Scanner (System.in);
        String theCommand;
        String[] theArrayOfComponentsOfTheCommand;
        
        while (true)
        {
            System.out.print ("--> ");
            theCommand = theScannerForACommand.nextLine();
            theArrayOfComponentsOfTheCommand = theCommand.split (" ", 2);
            
            if (theExitCommandIsRepresentedBy (theArrayOfComponentsOfTheCommand))
            {
                theScannerForACommand.close();
            }
            
            this.executesTheCommandRepresentedBy (theArrayOfComponentsOfTheCommand);
            
        }
        
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------------
     * theExitCommandIsRepresentedBy indicates whether or not the exit command is represented by an array of components of a
     * command.
     * 
     * @param theArrayOfComponentsOfTheCommand
     * @return
     -------------------------------------------------------------------------------------------------------------------- */
    
    private boolean theExitCommandIsRepresentedBy (String[] theArrayOfComponentsOfTheCommand)
    {
        if ((theArrayOfComponentsOfTheCommand.length == 1) &&
            (theArrayOfComponentsOfTheCommand[0].toLowerCase().equals ("exit")))
        {
            return true;
        }
        
        return false;
    }
    
}
