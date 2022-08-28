package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAParser;


import java.math.BigDecimal;
import java.util.Calendar;
import java.util.Date;
import org.json.JSONArray;
import org.json.JSONObject;


/** *************************************************************************************************************************
 * AParserForAJsonObjectRepresentingForecasts represents the structure for a parser for a JSON object representing forecasts,
 * which displays various forecasts corresponding to a JSON object.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/29/21
 ************************************************************************************************************************ */

public class AParserForAJsonObjectRepresentingForecasts {

    
    private JSONObject jsonObjectRepresentingForecasts;
    
    
    /** ----------------------------------------------------------------------------------------------------------------
     * AParserForAJsonObjectRepresentingForecasts(JSONObject theJsonObjectToUse) is the one-parameter constructor for
     * AParserForAJsonObjectRepresentingForecasts, which sets the JSON object representing forecasts of this parser to a
     * provided JSON object.
     * 
     * @param theJsonObjectToUse
     ---------------------------------------------------------------------------------------------------------------- */
    
    public AParserForAJsonObjectRepresentingForecasts (JSONObject theJsonObjectToUse)
    {
        this.jsonObjectRepresentingForecasts = theJsonObjectToUse;
    }
    
    
    /** ------------------------------------------------------
     * displaysTheCurrentWeather displays the current weather.
     ------------------------------------------------------ */
    
    public void displaysTheCurrentWeather()
        {
        
        JSONObject theJsonObjectRepresentingTheCurrentWeather =
            this.jsonObjectRepresentingForecasts.getJSONObject ("current");
        
        System.out.print(
            "Current Weather\n" +
            "The time is " + this.providesTheDateAndTimeCorrespondingTo (theJsonObjectRepresentingTheCurrentWeather.getLong ("dt")) + ".\n" +
            "The sunrise today occurs at " + this.providesTheDateAndTimeCorrespondingTo (theJsonObjectRepresentingTheCurrentWeather.getLong ("sunrise")) + ".\n" +
            String.format ("The temperature in degrees Celsius is %.1f", this.providesTheCelsiusTemperatureCorrespondingTo (theJsonObjectRepresentingTheCurrentWeather.getBigDecimal ("temp"))) + ".\n" +
            "The visibility in meters is " + theJsonObjectRepresentingTheCurrentWeather.getInt ("visibility") + ".\n" +
            "The ultraviolet index is " + theJsonObjectRepresentingTheCurrentWeather.getBigDecimal ("uvi") + ".\n\n"
        );
    }
    
    
    /** ----------------------------------------------------
     * displaysTheWeeklyForecast displays a weekly forecast.
     ---------------------------------------------------- */
    
    public void displaysTheWeeklyForecast()
    {
        System.out.println ("Weekly Forecast");
        
        JSONArray theJsonArrayOfDailyForecasts = this.jsonObjectRepresentingForecasts.getJSONArray ("daily");
        JSONObject theJsonObjectRepresentingADailyForecast;
        for (int i = 0; i < theJsonArrayOfDailyForecasts.length(); i++)
        {
            theJsonObjectRepresentingADailyForecast = theJsonArrayOfDailyForecasts.getJSONObject (i);
            Date theDateTimeForTheDailyForecast =
                this.providesTheDateAndTimeCorrespondingTo (theJsonObjectRepresentingADailyForecast.getLong ("dt"));

            System.out.print(
                "For " + this.providesTheRepresentationOfTheDateOf (theDateTimeForTheDailyForecast)  + ", " +
                "the probability of precipitation is " + theJsonObjectRepresentingADailyForecast.getBigDecimal ("pop") + ".\n"
            );
        }
        System.out.println();
    }
    
    
    /** ------------------------------------------------------------------------------------------
     * displaysTheProbabilityOfPrecipitationToday displays the probability of precipitation today.
     ------------------------------------------------------------------------------------------ */
    
    public void displaysTheProbabilityOfPrecipitationToday()
    {
        JSONArray theJsonArrayOfDailyForecasts = this.jsonObjectRepresentingForecasts.getJSONArray ("daily");
        JSONObject theJsonObjectRepresentingTodaysForecast = theJsonArrayOfDailyForecasts.getJSONObject (0);
        
        System.out.print(
            "The probability of precipitation today is " +
            theJsonObjectRepresentingTodaysForecast.getBigDecimal ("pop") + ".\n\n"
        );
    }
    
    
    /** --------------------------------------------------------------------------------------------------------------------
     * displaysTheAverageVolumePrecipitationPerUnitAreaForThisMinute provides the average volume precipitation per unit area
     * for this minute.
     -------------------------------------------------------------------------------------------------------------------- */
    
    public void displaysTheAverageVolumePrecipitationPerUnitAreaForThisMinute()
    {
        JSONArray theJsonArrayOfMinutelyForecasts = this.jsonObjectRepresentingForecasts.getJSONArray ("minutely");
        JSONObject theJsonObjectRepresentingThisMinutesForecast = theJsonArrayOfMinutelyForecasts.getJSONObject (0);
        
        System.out.print(
            "The average volume precipitation per unit area for this minute in millimeters was " +
            theJsonObjectRepresentingThisMinutesForecast.getBigDecimal ("precipitation") + ".\n\n"
        );
    }
    
        
    // TODO: Comment out this method after testing.
    protected JSONObject providesTheJsonObjectRepresentingForecasts()
    {
        return this.jsonObjectRepresentingForecasts;
    }
    
    
    /** ---------------------------------------------------------------------------------------------------------------
     * providesTheCelsiusTemperatureCorrespondingTo provides the Celsius temperature corresponding to a provided Kelvin
     * temperature.
     * 
     * @param theKelvinTemperature
     * @return
     --------------------------------------------------------------------------------------------------------------- */
    
    private double providesTheCelsiusTemperatureCorrespondingTo (BigDecimal theKelvinTemperature)
    {
        return theKelvinTemperature.doubleValue() - 273.15;
    }
    
    
    /** ------------------------------------------------------------------------------------------
     * providesTheDateTimeCorrespondingTo provides the date and time corresponding to a Unix time.
     * 
     * @param theUnixTimeInSeconds
     * @return
     ------------------------------------------------------------------------------------------ */
    
    private Date providesTheDateAndTimeCorrespondingTo (long theUnixTimeInSeconds)
    {
        return new Date (theUnixTimeInSeconds * 1000);
    }
    
    
    /** ----------------------------------------------------------------------------------------------------
     * providesTheRepresentationOfTheDateOf provides a representation of the date of a date-and-time object.
     * 
     * @param theDateTime
     * @return
     ---------------------------------------------------------------------------------------------------- */
    
    private String providesTheRepresentationOfTheDateOf (Date theDateAndTime)
    {
        Calendar theCalendar = Calendar.getInstance();
        theCalendar.setTime (theDateAndTime);
        int theMonth = theCalendar.get (Calendar.MONTH) + 1;
        int theDay = theCalendar.get (Calendar.DAY_OF_MONTH);
        int theYear = theCalendar.get (Calendar.YEAR);
        
        return theMonth + "/" + theDay + "/" + theYear;
    }
    
}