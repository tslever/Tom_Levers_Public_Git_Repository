package Com.TSL.TrafficLightSimulatorUtilities;


import Com.TSL.TrafficLightSimulatorUtilities.TrafficLightUtilities.ATrafficLight;
import javafx.scene.control.Button;


/** *****************************************************************************************************************
 * ATrafficLightStateAdvancingButton represents the structure for a button that advances the state of a traffic light
 * from green to yellow, yellow to red, and red to green.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/18/21
 **************************************************************************************************************** */

public class ATrafficLightStateAdvancingButton extends Button
{

    private ATrafficLight trafficLight;
    
    
    /** ------------------------------------------------------------------------------------------------------------
     * ATrafficLightStateAdvancingButton(ATrafficLight theTrafficLightToUse) is the one-parameter constructor for
     * ATrafficLightStateAdvancingButton, which sets the traffic light of this traffic light state advancing button,
     * sets the text of this button, and sets the event handler of this button.
     * 
     * @param theTrafficLightToUse
     ----------------------------------------------------------------------------------------------------------- */
    
    public ATrafficLightStateAdvancingButton (ATrafficLight theTrafficLightToUse)
    {
        
        this.trafficLight = theTrafficLightToUse;
        
        this.setText ("Advance the State of the Traffic Light");
        
        this.setOnAction (this.trafficLight::advancesItsTrafficLightColor);
        
    }
    
}
