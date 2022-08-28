package Com.TSL.UtilitiesForSimulatingADeckOfCards;


/** *******************************************************************************************************************
 * ADealFromAnEmptyDeckException represents the structure of an exception that occurs when a deal from an empty deck is
 * requested.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/13/21
 ****************************************************************************************************************** */

public class ADealFromAnEmptyDeckException extends Exception
{

    /** ---------------------------------------------------------------------------------------------------------------
     * ADealFromAnEmptyDeckException is the one-parameter constructor for ADealFromAnEmptyDeckException, which passes a
     * message to the one-parameter constructor of Exception.
     * 
     * @param message
     --------------------------------------------------------------------------------------------------------------- */
    
    public ADealFromAnEmptyDeckException(String message)
    {
        super(message);
    }
    
}
