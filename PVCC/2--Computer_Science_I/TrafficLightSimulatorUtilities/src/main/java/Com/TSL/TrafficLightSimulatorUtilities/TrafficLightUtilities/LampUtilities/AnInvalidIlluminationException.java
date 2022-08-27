package Com.TSL.TrafficLightSimulatorUtilities.TrafficLightUtilities.LampUtilities;


/** *****************************************************************************************************************
 * AnInvalidIlluminationException represents the structure for an exception that occurs when a lamp that is off tries
 * to turn itself off or a lamp that is on tries to turn itself on.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/18/21
 ***************************************************************************************************************** */

public class AnInvalidIlluminationException extends RuntimeException
{
    
    /** --------------------------------------------------------------------------------------------------------------
     * AnInvalidIlluminationException(String message) is the one-parameter constructor for
     * AnInvalidIlluminationException, which passes its argument to the one-parameter constructor of RuntimeException.
     * 
     * @param message
     -------------------------------------------------------------------------------------------------------------- */
    
    public AnInvalidIlluminationException (String message)
    {
        
        super (message);
        
    }
    
}
