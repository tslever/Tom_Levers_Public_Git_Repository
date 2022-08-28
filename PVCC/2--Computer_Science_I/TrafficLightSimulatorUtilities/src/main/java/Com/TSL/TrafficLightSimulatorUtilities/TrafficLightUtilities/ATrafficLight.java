package Com.TSL.TrafficLightSimulatorUtilities.TrafficLightUtilities;


import java.util.ArrayList;
import Com.TSL.TrafficLightSimulatorUtilities.TrafficLightUtilities.LampUtilities.Lamp;
import javafx.event.ActionEvent;
import javafx.scene.Group;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;


/** ******************************************************************
 * ATrafficLight represents the structure for a virtual traffic light.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/18/21
 ****************************************************************** */

public class ATrafficLight extends Group
{
    
    private ATrafficLightColor trafficLightColor;
    private ArrayList <Lamp> arrayListOfLamps;
    
    
    /** -------------------------------------------------------------------------------------------------------------
     * ATrafficLight() is the zero-parameter constructor for ATrafficLight, which creates a case, creates three lamps
     * and an array of the three lamps, adds the created shapes as children of the traffic light, initializes the
     * traffic light's color, and turns the corresponding lamp on.
     ------------------------------------------------------------------------------------------------------------- */
    
    public ATrafficLight (double theWidthOfItsParent)
    {
             
        double theWidthOfTheTrafficLight = theWidthOfItsParent / 4.0;
        double theHeightOfTheTrafficLight = theWidthOfTheTrafficLight * 2.0;      
        
        Rectangle theCase = new Rectangle (theWidthOfTheTrafficLight, theHeightOfTheTrafficLight, Color.GOLD);

        
        double halfTheWidthOfTheTrafficLight = theWidthOfTheTrafficLight / 2.0;
        double theDistanceBetweenTwoLamps = theWidthOfTheTrafficLight / 8.0;
        double theRadiusOfALamp = theWidthOfTheTrafficLight / 4.0;
        
        Lamp theGreenLamp = new Lamp (
            /*centerX = */ halfTheWidthOfTheTrafficLight,
            /*centerY = */ theHeightOfTheTrafficLight - theDistanceBetweenTwoLamps - theRadiusOfALamp,
            /*radius = */ theRadiusOfALamp,
            /*theColorToUseWhenOff = */ Color.DARKGREEN,
            /*theColorToUseWhenOn = */ Color.GREEN
        );
        
        Lamp theYellowLamp = new Lamp (
            /*centerX = */ halfTheWidthOfTheTrafficLight,
            /*centerY = */ theWidthOfTheTrafficLight,
            /*radius = */ theRadiusOfALamp,
            /*theColorToUseWhenOff = */ Color.DARKGOLDENROD,
            /*theColorToUseWhenOn = */ Color.YELLOW
        );
        
        Lamp theRedLamp = new Lamp (
            /*centerX = */ halfTheWidthOfTheTrafficLight,
            /*centerY = */ theDistanceBetweenTwoLamps + theRadiusOfALamp,
            /*radius = */ theRadiusOfALamp,
            /*theColorToUseWhenOff = */ Color.DARKRED,
            /*theColorToUseWhenOn = */ Color.RED
        );
        
        this.arrayListOfLamps = new ArrayList <Lamp>();
        this.arrayListOfLamps.add (theGreenLamp);
        this.arrayListOfLamps.add (theYellowLamp);
        this.arrayListOfLamps.add (theRedLamp);
        

        this.getChildren().addAll (theCase, theGreenLamp, theYellowLamp, theRedLamp);
        
        
        this.trafficLightColor = ATrafficLightColor.GREEN;
        this.arrayListOfLamps.get (this.trafficLightColor.ordinal()).turnsOn();
        
    }
    
    
    /** -----------------------------------------------------------------------------------
     * advancesItsTrafficLightColor advances the traffic-light color of this traffic light.
     ----------------------------------------------------------------------------------- */
    
    public void advancesItsTrafficLightColor (ActionEvent event)
    {
        
        this.arrayListOfLamps.get (this.trafficLightColor.ordinal()).turnsOff();
        
        this.trafficLightColor = ATrafficLightColor.values() [
            (this.trafficLightColor.ordinal() + 1) % ATrafficLightColor.values().length
        ];
     
        this.arrayListOfLamps.get (this.trafficLightColor.ordinal()).turnsOn();
        
    }
    
}
