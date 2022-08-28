package Com.TSL.TrafficLightSimulatorUtilities;


import javafx.geometry.Rectangle2D;
import javafx.stage.Screen;


/** ********************************************************************************************************************
 * TheConfigurations encapsulates definitions of actual parameters needed for the smooth functioning of the application.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/18/21
 ******************************************************************************************************************** */

public class TheConfigurations
{

    private static final Rectangle2D THE_WIDTH_AND_HEIGHT_OF_THE_SCREEN = Screen.getPrimary().getBounds();
    public static final double THE_WIDTH_OF_THE_APP = THE_WIDTH_AND_HEIGHT_OF_THE_SCREEN.getWidth() - 25.0;
    public static final double THE_HEIGHT_OF_THE_APP = THE_WIDTH_AND_HEIGHT_OF_THE_SCREEN.getHeight() - 75.0;
    
}
