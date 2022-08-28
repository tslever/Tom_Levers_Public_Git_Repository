package Com.TSL.FahrenheitConverterUtilities;


import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;


//************************************************************************
//  FahrenheitConverter.java       Author: Lewis/Loftus
//
//  Demonstrates the use of a TextField and a GridPane.
//************************************************************************

public class FahrenheitConverter extends Application
{
    //--------------------------------------------------------------------
    //  Launches the temperature converter application.
    //--------------------------------------------------------------------
	
    public void start(Stage primaryStage)
    {
    	FahrenheitPane fahrenheitPane = new FahrenheitPane();
    	fahrenheitPane.setPrefWidth(300);
    	
        Scene scene = new Scene(fahrenheitPane/*new FahrenheitPane()*/, 300, 150);
        
        primaryStage.setTitle("Fahrenheit Converter");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
    
    public static void main(String[] args)
    {
        launch(args);
    }
    
}