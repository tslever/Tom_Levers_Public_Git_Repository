package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAParser;


import Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnHttpRequestManager.AnHttpRequestManager;

import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.net.MalformedURLException;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;


/** ********************************************************************************************************
 * AParserTest encapsulates methods to test the functionality of AParserForAJsonObjectRepresentingForecasts.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/25/21
 ******************************************************************************************************* */

public class AParserTest
{

    /** -----------------------------------------------------------------------------
     * testAParser tests functionality of AParserForAJsonObjectRepresentingForecasts. 
     ----------------------------------------------------------------------------- */
    
    @Test
    public void testAParser()
    {
        try
        {
            AParserForAJsonObjectRepresentingForecasts theParserForAJsonObjectRepresentingForecasts =
                new AParserForAJsonObjectRepresentingForecasts(
                    new AnHttpRequestManager().providesAJsonObjectRepresentingForecastsBasedOn ("22903")
                );
            
            JSONObject theJsonObjectRepresentingForecasts =
                theParserForAJsonObjectRepresentingForecasts.providesTheJsonObjectRepresentingForecasts();
            
            System.out.print(
                "Displaying a JSON object representing forecasts.\n" + theJsonObjectRepresentingForecasts + "\n\n"
            );
            
            System.out.print ("Displaying derived forecasts for comparison.\n\n");
            theParserForAJsonObjectRepresentingForecasts.displaysTheCurrentWeather();
            theParserForAJsonObjectRepresentingForecasts.displaysTheWeeklyForecast();
            theParserForAJsonObjectRepresentingForecasts.displaysTheProbabilityOfPrecipitationToday();
            theParserForAJsonObjectRepresentingForecasts.displaysTheAverageVolumePrecipitationPerUnitAreaForThisMinute();
        }
        catch (MalformedURLException theMalformedUrlException)
        {
            fail (theMalformedUrlException.getMessage());
        }
        catch (IOException theIoException)
        {
            fail (theIoException.getMessage());
        }
    }
    
}