package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForADatabaseManager;


/** *****************************************************************************************************************
 * AnInvalidZipCodeException represents the structure for an invalid zip code exception, which occurs when a zip code
 * is evaluated to be invalid.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/28/21
 **************************************************************************************************************** */

public class AnInvalidZipCodeException extends Exception
{

    /** ---------------------------------------------------------------------------------------------------------------
     * AnInvalidZipCodeException (String message) is the one-parameter constructor for AnInvalidZipCodeException, which
     * passes a message to Exception's one-parameter constructor. 
     * 
     * @param message
     --------------------------------------------------------------------------------------------------------------- */
    
    public AnInvalidZipCodeException (String message)
    {
        super(message);
    }
    
}
