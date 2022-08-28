package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnHttpRequestManager;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import org.json.JSONObject;
import org.json.JSONTokener;


/** *************************************************************************************************
 * AnHttpRequestManager represents the structure for an HTTP-request manager, which retrieves weather
 * information from a web-based endpoint based on a zip code.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/29/21
 ************************************************************************************************ */

public class AnHttpRequestManager
{
    
    private final String API_KEY = "4a3cb03781d8f97e5d6969ad24786656";
    

    /** --------------------------------------------------------------------------------------------
     * providesAJsonObjectRepresentingForecastsBasedOn provides a JSON object representing forecasts
     * based on a zip code.
     * 
     * @param theZipCode
     * @return
     * @throws IOException
     * @throws MalformedURLException
     ------------------------------------------------------------------------------------------- */
    
    public JSONObject providesAJsonObjectRepresentingForecastsBasedOn (String theZipCode)
        throws IOException, MalformedURLException
    {
        AGeographicPosition theGeographicPosition = this.providesTheGeographicPositionCorrespondingTo (theZipCode);
        
        String theUrlOfTheForecastingEndpointAsAString =
            "https://api.openweathermap.org/data/2.5/onecall?lat=" + theGeographicPosition.providesItsLatitude() +
            "&lon=" + theGeographicPosition.providesItsLongitude() + "&appid=" + this.API_KEY;
        
        return this.providesTheJsonObjectCorrespondingTo (theUrlOfTheForecastingEndpointAsAString);
    }
    
    
    /** ---------------------------------------------------------------------------------------------
     * providesTheGeographicPositionCorrespondingTo provides the geographic position corresponding to
     * a zip code.
     * 
     * @param theZipCode
     * @return
     * @throws IOException
     * @throws MalformedURLException
     --------------------------------------------------------------------------------------------- */
    
    // TODO: Change to private after testing.
    protected AGeographicPosition providesTheGeographicPositionCorrespondingTo (String theZipCode)
        throws IOException, MalformedURLException
    {
        String theUrlOfTheGeocodingEndpointAsAString = 
            "http://api.openweathermap.org/geo/1.0/zip?zip=" + theZipCode + "&appid=" + this.API_KEY;
        
        JSONObject theGeographicJsonObject =
            providesTheJsonObjectCorrespondingTo (theUrlOfTheGeocodingEndpointAsAString);
        
        String theLatitudeAsAString = theGeographicJsonObject.getBigDecimal ("lat").toString();
        
        String theLongitudeAsAString = theGeographicJsonObject.getBigDecimal ("lon").toString();
        
        double theLatitude = Double.parseDouble (theLatitudeAsAString);
        
        double theLongitude = Double.parseDouble (theLongitudeAsAString);
        
        return new AGeographicPosition (theLatitude, theLongitude);
    }
    
    
    /** -----------------------------------------------------------------------------------------
     * providesTheJsonObjectCorrespondingTo provides the JSON object corresponding to a URL of an
     * web-based endpoint that provides weather information.
     * 
     * @param theUrlOfTheEndpointAsAString
     * @return
     * @throws IOException
     * @throws MalformedURLException
     ---------------------------------------------------------------------------------------- */
    
    private JSONObject providesTheJsonObjectCorrespondingTo (String theUrlOfTheEndpointAsAString)
        throws IOException, MalformedURLException
    {
        URL theUrlOfTheEndpoint = new URL (theUrlOfTheEndpointAsAString); // throws MalformedURLException
        
        URLConnection theUrlConnection = theUrlOfTheEndpoint.openConnection(); // throws IOException
        
        InputStream theInputStream = theUrlConnection.getInputStream(); // throws IOException
        
        InputStreamReader theInputStreamReader = new InputStreamReader (theInputStream);
        
        BufferedReader theBufferedReader = new BufferedReader (theInputStreamReader);
        
        JSONTokener theJsonTokener = new JSONTokener (theBufferedReader);
        
        return new JSONObject (theJsonTokener);
    }
	
}