package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnHttpRequestManager;


/** *******************************************************************************************************************
 * AGeographicPosition represents the structure for a geographic position, which encapsulates functionality relating to
 * latitude and longitude.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/29/21
 ****************************************************************************************************************** */

public class AGeographicPosition
{
    
    private double latitude;
    private double longitude;
    
    
    /** -------------------------------------------------------------------------------------------------------------------
     * AGeographicPosition(double theLatitudeToUse, double theLongitudeToUse) is the two-parameter constructor for
     * AGeographicPosition, which sets the latitude of this position to a provided latitude, and sets the longitude of this
     * position to a provided longitude.
     * 
     * @param theLatitudeToUse
     * @param theLongitudeToUse
     ------------------------------------------------------------------------------------------------------------------ */

    public AGeographicPosition (double theLatitudeToUse, double theLongitudeToUse)
    {
        this.latitude = theLatitudeToUse;
        this.longitude = theLongitudeToUse;
    }
    
    
    /** ----------------------------------------------------------
     * providesItsLatitude provides the latitude of this position.
     * 
     * @return
     ---------------------------------------------------------- */
    
    public double providesItsLatitude()
    {
        return this.latitude;
    }
    
    
    /** ------------------------------------------------------------
     * providesItsLongitude provides the longitude of this position.
     * 
     * @return
     ------------------------------------------------------------ */
    
    public double providesItsLongitude()
    {
        return this.longitude;
    }
    
    
    /** --------------------------------------------------------------
     * toString provides a coordinate representation of this position.
     * 
     * @return
     -------------------------------------------------------------- */
    
    @Override
    public String toString()
    {
        return "(" + this.latitude + ", " + this.longitude + ")";
    }

}
