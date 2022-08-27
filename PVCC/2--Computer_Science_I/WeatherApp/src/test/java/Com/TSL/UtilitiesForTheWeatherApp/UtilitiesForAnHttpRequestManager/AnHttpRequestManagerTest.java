package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnHttpRequestManager;


import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.net.MalformedURLException;
import org.junit.jupiter.api.Test;


/** ***********************************************************************************************
 * AnHttpRequestManagerTest encapsulates methods to test the functionality of AnHttpRequestManager.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/25/21
 *********************************************************************************************** */

public class AnHttpRequestManagerTest
{

    /** --------------------------------------------------------------------
     * testAnHttpRequestManager tests functionality of AnHttpRequestManager. 
     -------------------------------------------------------------------- */
    
    @Test
    public void testAnHttpRequestManager()
    {

        AnHttpRequestManager theHttpRequestManager = new AnHttpRequestManager();
        
        try
        {
            AGeographicPosition theGeographicPosition =
                theHttpRequestManager.providesTheGeographicPositionCorrespondingTo ("22903");
            
            System.out.print(
                "The latitude and longitude of Charlottesville, VA, USA is " + theGeographicPosition + ".\n\n"
            );
            
            theGeographicPosition = theHttpRequestManager.providesTheGeographicPositionCorrespondingTo ("19446");
            
            System.out.print(
                "The latitude and longitude of Lansdale, PA, USA is " + theGeographicPosition + ".\n\n"
            );
            
        }
        catch (MalformedURLException theMalformedUrlException)
        {
            fail (theMalformedUrlException.getMessage());
        }
        catch (IOException theIoException)
        {
            fail (theIoException.getMessage());
        }
        
        try
        {
            System.out.println ("Trying to get the geographic position for an invalid zip code.");
            theHttpRequestManager.providesTheGeographicPositionCorrespondingTo ("-----");
        }
        catch (MalformedURLException theMalformedUrlException)
        {
            System.out.println ("Malformed URL Exception: " + theMalformedUrlException.getMessage());
        }
        catch (IOException theIoException)
        {
            System.out.println ("IO Exception: " + theIoException.getMessage());
        }
        
    }
    
}