package Com.TSL.TrafficLightSimulatorUtilities;


import Com.TSL.TrafficLightSimulatorUtilities.TrafficLightUtilities.ATrafficLight;
import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.layout.FlowPane;
import javafx.scene.Scene;
import javafx.stage.Stage;


/** **************************************************************************************************************
 * TrafficLightSimulator encapsulates the entry point of this program, which launches a graphical application that
 * allows a user to press a button that advances the state of a virtual traffic light.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/18/21
 ************************************************************************************************************** */

public class TheTrafficLightSimulator extends Application
{
	
    /** ------------------------------------------------------------------------------
     * start mobilizes to a stage components that interact with a user and each other.
     ------------------------------------------------------------------------------ */

    @Override
    public void start (Stage stage)
    {

        ATrafficLight theTrafficLight = new ATrafficLight (TheConfigurations.THE_WIDTH_OF_THE_APP);
        FlowPane theFlowPane = new FlowPane (theTrafficLight, new ATrafficLightStateAdvancingButton(theTrafficLight));
        theFlowPane.setAlignment (Pos.CENTER);
        theFlowPane.setHgap (20);

        Scene scene =
            new Scene (theFlowPane, TheConfigurations.THE_WIDTH_OF_THE_APP, TheConfigurations.THE_HEIGHT_OF_THE_APP);    	

        stage.setTitle ("Traffic Light Simulator");
        stage.setResizable (false);
        stage.setScene (scene);
        stage.show();
        
    }


    /** ------------------------------------------------------------------------------------------------------------
     * main is the entry point of this application, which allows a user to press a button that advances the state of
     * a virtual traffic light.
     * 
     * @param args
    ------------------------------------------------------------------------------------------------------------ */

    public static void main (String[] args)
    {

        launch();

    }

}