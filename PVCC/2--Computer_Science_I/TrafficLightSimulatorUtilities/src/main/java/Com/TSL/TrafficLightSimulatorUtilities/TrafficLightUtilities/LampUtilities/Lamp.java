package Com.TSL.TrafficLightSimulatorUtilities.TrafficLightUtilities.LampUtilities;


import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;


/** ***********************************************************************************************************
 * Lamp represents the structure for a lamp of a traffic light that has the capability to turn off and turn on.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/18/21
 *********************************************************************************************************** */

public class Lamp extends Circle
{
    
    private Color colorWhenOff;
    private Color colorWhenOn;
    private AnIllumination illumination;
    
    
    /** -----------------------------------------------------------------------------------------------------------------
     * Lamp(double centerX, double centerY, double radius, Color theColorToUseWhenOff, Color theColorToUseWhenOn) is the
     * five-parameter constructor for Lamp, which passes horizontal-position, vertical-position, radius, and fill-color
     * arguments to the four-parameter constructor of Circle, and sets this lamp's off color, on color, and illumination.
     * 
     * @param centerX
     * @param centerY
     * @param radius
     * @param theColorToUseWhenOff
     * @param theColorToUseWhenOn
     ----------------------------------------------------------------------------------------------------------------- */
    
    public Lamp (double centerX, double centerY, double radius, Color theColorToUseWhenOff, Color theColorToUseWhenOn)
    {
        
        super (centerX, centerY, radius, theColorToUseWhenOff);
        
        this.colorWhenOff = theColorToUseWhenOff;
        this.colorWhenOn = theColorToUseWhenOn;
        this.illumination = AnIllumination.OFF;
        
    }
    
    
    /** --------------------------------------------------------------------------------
     * turnsOff turns this lamp off by changing this lamp's illumination and fill color.
     -------------------------------------------------------------------------------- */
    
    public void turnsOff()
    {
        
        if (this.illumination == AnIllumination.OFF) {
            throw new AnInvalidIlluminationException ("A lamp that was off tried to turn off.");
        }
        
        this.illumination = AnIllumination.OFF;
        this.setFill (this.colorWhenOff);
        
    }
    
    
    /** ------------------------------------------------------------------------------
     * turnsOn turns this lamp on by changing this lamp's illumination and fill color.
     ------------------------------------------------------------------------------ */
    
    public void turnsOn()
    {
        
        if (this.illumination == AnIllumination.ON) {
            throw new AnInvalidIlluminationException ("A lamp that was on tried to turn on.");
        }
        
        this.illumination = AnIllumination.ON;
        this.setFill (this.colorWhenOn);
        
    }
    
}
